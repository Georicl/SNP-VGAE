#!/usr/bin/perl
$threshold = 1;

$dir = ".";
$from = "^Correlation.";

opendir(DIR, "$dir");
@files=grep(/$from/, readdir(DIR));
closedir(DIR);

die "No files matching pattern: \"$from\" found in directory: \"$dir\"\n" 
unless @files;
foreach $file (@files) {
#warn "Processing $file\n";
open (FILE, $file) || die "Can't open file $file\n";

$head = <FILE>;
chomp($head);
@data = split (/\t/, $head);
for ($n =0; $n <=$#data; $n++) {
$Names->{$n} = $data[$n];
$Columns->{$data[$n]} = $n;
$phenotype = $data[0];
}


while (<FILE>) {
chomp;
@data = split;
for ($n = 1; $n <=$#data; $n++) {
$Correlations->{$phenotype}->{$Names->{$n}}->{$data[0]} = $data[$n];
#print "$phenotype $Names->{$n}\n";
#print "$data[0]\n";
}

}


}

foreach $phenotype1 (sort keys %{$Correlations}) {

foreach  $phenotype2 (sort keys %{$Correlations->{$phenotype1}} ) {
if ($Correlations->{$phenotype1}->{$phenotype2}->{'logP'} >= $threshold) {
$logP = $Correlations->{$phenotype1}->{$phenotype2}->{'logP'} ;
$corr =  $Correlations->{$phenotype1}->{$phenotype2}->{'corr'} ;
$df = $Correlations->{$phenotype1}->{$phenotype2}->{'df'};
$statistic = $Correlations->{$phenotype1}->{$phenotype2}->{'statistic'}; 

printf "$phenotype1\t$phenotype2\t%3.3f\t%3.2f\t%4d\n", $corr, $logP, $df;

}
}

}
