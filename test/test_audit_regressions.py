"""Small CPU regressions for the public code audit; no external data needed."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from model.vgae import VGAEModel
from graph.build_graph import GraphBuilder
from train.trainer import VGAETrainer
from train.cross_validation import run_kfold_cv

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'test' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_problem():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 3, (5, 9), dtype=np.int32)
    grm = np.full((9, 9), .1, dtype=np.float32) + np.eye(9, dtype=np.float32)
    y = np.linspace(-1, 1, 9, dtype=np.float32)
    return g, grm, y


def make_trainer():
    g, grm, y = tiny_problem()
    builder = GraphBuilder()
    adj, raw = builder.knn_graph_builder(grm, 2)
    features = builder.build_node_features(g)
    masks = [torch.arange(9) // 3 == i for i in range(3)]
    model = VGAEModel(d_snp=5, d_hidden=4, d_z=2, mlp_hidden=4, dropout=.1)
    return VGAETrainer(model, adj, raw, features, torch.tensor(y), *masks,
                       beta=1., max_epochs=3, kl_warmup_epochs=2, progress=False)


def test_eval_uses_latent_mean_and_is_repeatable():
    trainer = make_trainer()
    model = trainer.model.eval()
    a = model(trainer.node_features, trainer.adj_norm)
    b = model(trainer.node_features, trainer.adj_norm)
    torch.testing.assert_close(a['z'], a['mu'], rtol=0, atol=0)
    torch.testing.assert_close(a['y_pred'], b['y_pred'], rtol=0, atol=0)
    model.train()
    assert not torch.equal(model.reparameterize(a['mu'], a['logvar']), a['mu'])


def test_validation_fixed_beta_edges_and_does_not_consume_rng():
    trainer = make_trainer()
    before = torch.get_rng_state().clone()
    a = trainer._validate()
    b = trainer._validate()
    assert a == b
    assert torch.equal(before, torch.get_rng_state())
    assert a['total'] == pytest.approx(a['pheno'] + trainer.alpha*a['edge'] + trainer.beta*a['kl'])


def test_blup_scripts_tune_on_disjoint_sets(monkeypatch):
    # Spy on both lambda tuning and final refitting.
    masks = tuple(np.arange(9)//3 == i for i in range(3))
    for name in ['evaluate', 'statistical_validation']:
        script = load_script(name)
        calls = []
        def predict(grm, labels, train, target, lam):
            calls.append((np.asarray(train, bool), np.asarray(target, bool)))
            return np.arange(np.count_nonzero(target), dtype=float)
        monkeypatch.setattr(script, 'blup_predict', predict)
        import train.blup as blup
        monkeypatch.setattr(blup, 'blup_predict', predict)
        if name == 'evaluate':
            script.blup_cv(np.eye(9), np.arange(9.), [masks])
        else:
            monkeypatch.setattr(script, 'generate_kfold_masks', lambda *args: [masks])
            script.blup_cv(np.eye(9), np.arange(9.), 42)
        assert len(calls) == len(script.LAMBDA_GRID) + 1
        assert all(not np.any(train & target) for train, target in calls)
        assert np.array_equal(calls[-1][0], masks[0] | masks[1])


def test_blup_rejects_overlap_empty_and_invalid_lambda():
    from train.blup import select_blup_lambda
    tr = np.array([1,1,0,0], bool)
    va = ~tr
    for train, val, grid in [(tr,tr,[1.]), (tr,va,[]), (tr,va,[0.]), (tr,np.zeros(4,bool),[1.])]:
        with pytest.raises(ValueError):
            select_blup_lambda(np.eye(4), np.arange(4.), train, val, grid)


def test_pretrain_checkpoint_uses_fixed_target_beta(monkeypatch):
    import pretrain.train as pretrain
    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, x):
            return self.weight, self.weight, self.weight
    sequence = iter([(1., 10.), (.5, 1.), (2., 0.)])
    def loss(pred, x, mu, logvar, beta):
        recon, kl = next(sequence)
        total = pred-pred.detach()+recon+beta*kl
        return total, torch.tensor(recon), torch.tensor(kl)
    monkeypatch.setattr(pretrain, 'SNPVAE', TinyModel)
    monkeypatch.setattr(pretrain, 'vae_loss', loss)
    monkeypatch.setattr(pretrain, 'extract_embeddings', lambda model, *args, **kwargs: model.weight.detach().numpy())
    model, _ = pretrain.train_snp_vae(np.array([[0,1,2],[2,1,0]]), beta=1., lr=.01,
                                   epochs=3, batch_size=4, kl_warmup_epochs=2,
                                   patience=10, progress=False)
    # Fixed beta scores [11,1.5,2] choose epoch 2. Warmup totals [1,1,2] choose epoch 1.
    assert model.weight.item() == pytest.approx(-.02, abs=1e-6)


def test_kfold_seed_reproduces_small_training():
    g, grm, labels = tiny_problem()
    kwargs = dict(genotype=g, snp_embeddings=None, grm=grm, labels=labels,
                  k_folds=3, k_neighbors=2, d_hidden=4, d_hidden2=4, d_z=2,
                  mlp_hidden=4, max_epochs=2, patience=2, seed=123, progress=False)
    a = run_kfold_cv(**kwargs)
    b = run_kfold_cv(**kwargs)
    assert a['fold_metrics'] == b['fold_metrics']
    for model_a, model_b in zip(a['models'], b['models']):
        for key, value in model_a.state_dict().items():
            torch.testing.assert_close(value, model_b.state_dict()[key], rtol=0, atol=0)


def test_trainer_restores_best_fixed_criterion_checkpoint(monkeypatch):
    trainer = make_trainer()
    parameter = next(trainer.model.parameters())
    epoch = [0]
    scores = iter([3., 1., 2.])
    def train_epoch(beta):
        epoch[0] += 1
        with torch.no_grad():
            parameter.fill_(epoch[0])
        return dict(total=0., pheno=0., edge=0., kl=0.)
    def validate():
        return dict(total=next(scores), pheno=0., edge=0., kl=0.)
    monkeypatch.setattr(trainer, '_train_epoch', train_epoch)
    monkeypatch.setattr(trainer, '_validate', validate)
    monkeypatch.setattr(trainer, '_evaluate', lambda mask: dict(mse=0.,mae=0.,r=0.,r2=0.))
    result = trainer.train()
    assert result['best_epoch'] == 2
    assert torch.all(next(trainer.model.parameters()) == 2)


def test_statistical_script_actual_permutation_calls_match(monkeypatch, tmp_path, capsys):
    script = load_script('statistical_validation')
    g, grm, y = tiny_problem()
    monkeypatch.setattr(script, 'RESULT_DIR', str(tmp_path))
    monkeypatch.setattr(script, 'N_REPEATS', 2)
    monkeypatch.setattr(script, 'N_PERM_BLUP', 2)
    monkeypatch.setattr(script, 'N_PERM_VGAE', 2)
    monkeypatch.setattr(script, 'genotype_read', lambda *args: g)
    monkeypatch.setattr(script, 'grm_reader', lambda *args: grm)
    monkeypatch.setattr(script, 'grm_id_reader', lambda *args: [('f',str(i)) for i in range(9)])
    monkeypatch.setattr(script, 'load_phenotype', lambda *args: y)
    monkeypatch.setattr(script, 'train_snp_vae', lambda **kwargs: (None, np.ones((5,2))))
    monkeypatch.setattr(script, 'blup_cv', lambda *args, **kwargs: .1)
    calls = []
    def vgae(g, e, grm, labels, seed, config, device, progress=False):
        calls.append((dict(config), seed))
        return .2 + len(calls)*.01
    monkeypatch.setattr(script, 'vgae_cv', vgae)
    script.main()
    assert len(calls) == 4
    assert all(config == calls[0][0] for config, _ in calls)
    assert calls[0][1] == calls[-1][1] == 42
    output = capsys.readouterr().out
    assert '95%CI' not in output and '***' not in output
    assert '不作显著性判定' in output
