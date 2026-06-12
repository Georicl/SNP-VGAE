#!/usr/bin/perl
open (FILE, "CombinedPhenotypes") || die "Can't open file CombinedPhenotype\n";
$head = <FILE>;
chomp($head);
@data = split (/\t/, $head);
for ($n =0; $n <=$#data; $n++) {
$Names->{$n} = $data[$n];
$Columns->{$data[$n]} = $n;
}


while (<FILE>) {
chomp;
@data = split;
for ($n = 1; $n <=$#data; $n++) {
$Phenotypes->{$Names->{$n}}->{$data[$Columns->{'SUBJECT.NAME'}]} = $data[$n];
}
}

foreach $phenotype1 (sort keys %{$Phenotypes}) {

$outname = $phenotype1 . ".fit.txt";
warn "Processing $phenotype1 into $outname\n";

	open (OUT, ">$outname") || die "Can't open $outname\n";
	
	print OUT "SUBJECT.NAME\t$phenotype1";
	foreach $phenotype2 (sort keys %{$Phenotypes}) {
			unless ($phenotype1 eq $phenotype2) {
				print OUT "\t$phenotype2";		
			}
	}
 	print OUT "\n";
	foreach $id (sort keys %{$Phenotypes->{$phenotype1}}) {
		print OUT "$id\t$Phenotypes->{$phenotype1}->{$id}";
		foreach $phenotype2 (sort keys %{$Phenotypes}) {                
                        unless ($phenotype1 eq $phenotype2) {
			print OUT "\t$Phenotypes->{$phenotype2}->{$id}";
                        }
                        }
	print OUT"\n";
}
close (OUT);

open (ROUT, ">tmp.R") || die "Can't write to tmp.R\n";
warn "Writing tmp.R for $outname\n";

print ROUT qq{data<-read.delim(\"$outname\") 

anova(lm(phenotype ~ x + I(x^2) + I(x^3) + I(x^4), data=data))

results.mat <- matrix(ncol=dim(data)[2]-1, nrow=4)
colnames(results.mat) <- c(names(data)[-1])
results.mat[1,1] = "corr";
results.mat[2,1] = "statistic";
results.mat[3,1] = "df";
results.mat[4,1] = "logP";

for ( i in 3:dim(data)[2]) {
res<-cor.test(data[,2], data[,i])
#results.mat[1,i-1] <-res\$statistic
results.mat[1,i-1] <-res\$estimate
results.mat[2,i-1] <-res\$statistic
results.mat[3,i-1] <-res\$parameter
results.mat[4,i-1] <- -log10(res\$p.value) 

}
filename = paste (\"Correlation.\", names(data)[2], sep=\"\")
write.table(results.mat, file = filename, row.names = F, sep = \"\\t\", quote = F)

};

close (ROUT);

$command = "R --vanilla < tmp.R";
SpawnProcess ($command);
unlink($outname);

}


sub SpawnProcess {

    local($command)=@_;
 #  warn "$command\n";

   open(PROCESS, "$command 2>&1 |")
        || warn "$prog: spawn couldn't open \"$command\": $!\n"
            && return;

    my(@output)=<PROCESS>;

    close (PROCESS);
    $status=($? >> 8);

    if ( $status != 0 ) {
        warn "$prog: spawn \"$command\" returned non-zero exit status: @output\n";
    }

    return (@output);

}

