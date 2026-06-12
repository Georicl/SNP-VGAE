#!/usr/bin/perl
use strict;

my $file1 = $ARGV[0];
my $file2 = $ARGV[1];
my $Ids = {};
my $Phenotypes = {};
my $Filenames = {};
my $IdOrder = {};

open (PHEN, "hipp.phenotype.names") || die "Can't open phenotype.names - list of phenotypes\n";
while (<PHEN>) {
  chomp;
  my @data = split;
  my $file = $data[0] . ".resid";
  $Filenames->{$file} = $data[0];
}

open (FILE1, $file1) || die "Can't open id file $file1\n";

while (<FILE1>) {
  chomp;
  my @data = split;
  my $id = $data[0];
  $Ids->{$id} = $data[1];
  $IdOrder->{$data[1]} = $data[0];
}




foreach my $file (sort keys  %{$Filenames}) {
  my $phenotype = $Filenames->{$file};
  if (-e $file) {
    open (FILE, $file) || die "Can't open $file\n";
    my $head = <FILE>;
    while (<FILE>) {
      chomp;
      my @data = split;
      my $id = $data[0];
      if (exists($Ids->{$id})) {
	$Phenotypes->{$phenotype}{$id} = $data[1];
      }
    }
  }
  else {
    warn "Can't find file $file\n";
  }
}
print "SUBJECT.NAME";
foreach my $phenotype (sort keys   %{$Phenotypes}) {
print "\t$phenotype";
}
print "\n";
foreach my $pos (sort {$a <=> $b} keys  %{$IdOrder}) {
  my $id = $IdOrder->{$pos};
  print "$id";
  foreach my $phenotype (sort keys   %{$Phenotypes}) {
    if (exists($Phenotypes->{$phenotype}{$id})) {
      print "\t$Phenotypes->{$phenotype}{$id}";
    }
    else {
      print "\tNA";
    }
  }
print "\n";
}
