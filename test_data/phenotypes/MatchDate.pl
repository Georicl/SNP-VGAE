#!/usr/bin/perl

$LookUp = "LunarCycle.LookUp.txt";
open (LOOKUP, $LookUp)|| die "Can't open look up file  $LookUp\n";

my $head = <LOOKUP>;
chomp($head);
my @data = split (/\t/, $head);
for ($n = 0; $n <= $#data; $n++) {
$LColumns->{$data[$n]} = $n;
}

unless (exists($LColumns->{'Date.StudyDay'})) {
die "Can't find a column header Date.StudyDay in the lookup file\n";
}
while (<LOOKUP>) {
chomp;
my @data = split;

$studydate =  $data[$LColumns->{'Date.StudyDay'}];
#print "$studydate\n";
$StudyDate->{$studydate}->{'Phase'} = $data[$LColumns->{'Phase'}];
$StudyDate->{$studydate}->{'LunarProgress'} = $data[$LColumns->{'Lunar.Progress'}];
$StudyDate->{$studydate}->{'FullMoonExtent'} = $data[$LColumns->{'Full.Moon.Extent'}];
$StudyDate->{$studydate}->{'LunarDay'} = $data[$LColumns->{'Lunar.Day'}];
}

my $head = <>;
chomp($head);
$head =~ s/\r//;
my @data = split (/\t/, $head);
for ($n = 0; $n <= $#data; $n++) {
$Columns->{$data[$n]} = $n;
}


unless (exists($Columns->{'Date.StudyDay'})) {
die "Can't find a column header 'Date.StudyDay' \n";
}

print "$head\tLunar.Progress\tFull.Moon.Extent\tLunar.Day\tPhase\n";
while (<>) {

chomp;
s/\r//g;

my @data = split (/\t/, $_);

$subject = $data[$Columns->{'SUBJECT.NAME'}];
$studydate = $data[$Columns->{'Date.StudyDay'}];
if (exists($StudyDate->{$studydate})) {
print "$_\t$StudyDate->{$studydate}->{'LunarProgress'}\t$StudyDate->{$studydate}->{'FullMoonExtent'}\t$StudyDate->{$studydate}->{'LunarDay'}\t$StudyDate->{$studydate}->{'Phase'}\n";
}

}
