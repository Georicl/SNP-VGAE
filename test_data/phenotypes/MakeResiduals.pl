#!/usr/bin/perl
use strict;

my $modelmenu = "/groups/flint/ILLUMINA/ANALYSIS/ModelMenu080108.txt";
my $list = $ARGV[0];
my $PhenotypeDir = "/groups/flint/ILLUMINA/PHENOTYPES";
my $Menu = {};
my $Phenotypes ={};

open (MENU, $modelmenu) || die "Can't open model menu file $modelmenu\n";


while (<MENU>) {
chomp;
my @data = split ("\t", $_);
my $phenotype = $data[1];
my $phenotypefile = $data[0];
my $formula = $data[5];
$Menu->{$phenotype}{file} = $phenotypefile;
$Menu->{$phenotype}{formula} = $formula;
#print "$phenotype, $phenotypefile\n";
}

open (LIST, $list) || die "Can't open list file $list\n";
while (<LIST>) {
chomp;
my @data = split;
$Phenotypes->{$data[0]}++;
}

foreach my $phenotype (sort keys %{$Phenotypes}) {
unless (exists($Menu->{$phenotype})) {
die "Can't find $phenotype in the menu file\n";
}
my $phenotypefile = $Menu->{$phenotype}{file};
my $formula = $Menu->{$phenotype}{formula};
my $outfile = $phenotype . ".resid";
open (RFILE, ">file.R") || die "Can't write to Rfile \n";
print RFILE "source(\"/home/valdar/cvs/stable/hs/bagphenotype/startup.R\")
#bag.startup()
data<-read.delim(\"$PhenotypeDir/$phenotypefile\")
fit <-data.frame(SUBJECT.NAME = rep(NA, length(data\$SUBJECT.NAME)), Phenotype = rep (NA, length(data\$SUBJECT.NAME)))
fit\$Phenotype<-resid(lm($formula, data = data, na.action =na.exclude))
fit\$SUBJECT.NAME <- data\$SUBJECT.NAME
write.table(fit, file = \"$outfile\", sep = \"\\t\", quote = F, row.names = F)
\n";
my $command = "R --vanilla < file.R";
system($command);
}

