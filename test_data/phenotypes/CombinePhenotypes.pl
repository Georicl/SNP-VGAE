#!/usr/bin/perl

open (IDS, "ids") || die "Can't open ids file ids\n";

while (<IDS>) {
  chomp;
  ($id) = /^(A.*)/;
  $Ids->{$id}++;
}

open (PHEN, "test")|| die "Can't open phenotypes\n";
while (<PHEN>) {
  chomp;
  if (($phen) = /^(\S+)/) {
    $Phenotypes->{$phen}++;
  }
}

open (LIST, "files") || die "Can't open list of file names to process\n";

while (<LIST>) {
  chomp;
  if (($file) = /^(\S+)/) { 
    push (@files, $file) ;
  }
}

my $CombinedPhenotypes = {};

foreach my $file (@files) {
#  print "Processing $file\n";
  open (FILE, $file) || die "Can't find file $file\n";
  my $Columns = {};
  my $GoodPhenotypes = {};
  my $count = 0;
  my $head = <FILE>;
  chomp($head);
  
  my @data = split(/\t/,$head);
  for (my $n = 0; $n <=$#data; $n++) {
    $Columns->{$data[$n]} = $n;
  } 
  
  foreach $phenotype (keys %{$Phenotypes}) {
    if (exists($Columns->{$phenotype}) ) {
    #  print "found good phenotype $phenotype \n";
      $GoodPhenotypes->{$phenotype}++;
      $count++;
    }
  }
  
  if ($count == 0) {
    warn "NO USEFUL PHENOTYPES FOUND IN $file\n";
    next;
  }
  else {
    while (<FILE>) {
      chomp;
      my @data = split(/\t/, $_);
      foreach my $goodphenotype (keys %{$GoodPhenotypes}) {
	my $position = $Columns->{$goodphenotype};
	if (exists($data[$Columns->{$goodphenotype}])) {	
	  $CombinedPhenotypes->{$goodphenotype}->{$data[$Columns->{'SUBJECT.NAME'}]} = $data[$Columns->{$goodphenotype}];
	}
      }
      
    }
  }
}

foreach $phenotype (sort keys %{$CombinedPhenotypes}) {

	foreach $id  (sort keys %{$Ids}) {

	if ($CombinedPhenotypes->{$phenotype}->{$id} =~ /\S+/) {
	$Ready->{$id}->{$phenotype} = $CombinedPhenotypes->{$phenotype}->{$id};
	}

	else {
	$Ready->{$id}->{$phenotype} = "NA";			
	}
    }
}


print "SUBJECT.NAME";

foreach $phenotype (sort keys %{$CombinedPhenotypes}) {
	print "\t$phenotype";
	}
print "\n";


foreach $id (sort keys %{$Ready}) {
	print "$id";
	foreach $phenotype (sort keys %{$CombinedPhenotypes}) {
		print "\t$Ready->{$id}->{$phenotype}";
	}
	print "\n";
}
