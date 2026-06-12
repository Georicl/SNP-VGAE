#!/usr/bin/perl
$head = <>;
chomp($head);
@data = split(/\t/, $head) ;
for ($n =0; $n <=$#data; $n++) {
$Columns->{$data[$n]} = $n;
}
unless (exists( $Columns->{Covariates}) && exists($Columns->{Phenotype}) && exists ($Columns->{Model}) ) {
die "Incorrect column heads in $head\n";
}

print "PhenotypeFile\tPhenotype\tResponses\tCovariates\tSubset\tModel\tFunction\tFunctionOptions\tNullSimulationCovariates\tNullSimulationModel\tNullSimulationFunction\tNullSimulationFunctionOptions\n";


while (<>) {
chomp;
s/\"//g;
@data = split (/\t/, $_);
$phenotype = $data[$Columns->{'Phenotype'}];
$covariates = $data[$Columns->{'Covariates'}];  
$model = $data[$Columns->{'Model'}];
$function = "linear";
$phenotypefile = $data[$Columns->{'PhenotypeFile'}];
#$phenotypefile = "Fecal.txt";
$responses = $phenotype;
$ullSimulationCovariates = "Family," . $covariates;
$subset = "";
$functionoptions = "reduce.dmat=TRUE";
$nullsimulationcovariates = "Family," . $covariates;
$nullsimulationmodel =  $model . "  + (1|Family)";
$nullsimulationfunction = $function;
$nullsimulationfunctionoptions = "reduce.dmat=TRUE";

$Data->{$phenotypefile}->{$phenotype} =  "$phenotypefile\t$phenotype\t$phenotype\t$covariates\t$subset\t$model\t$function\t$functionoptions\t$nullsimulationcovariates\t$nullsimulationmodel\t$nullsimulationfunction\t$nullsimulationfunctionoptions";
}


foreach $file (sort keys %{$Data}) {
foreach $phenotype (sort keys %{$Data->{$file}}) {
print "$Data->{$file}->{$phenotype}\n";
}
}
