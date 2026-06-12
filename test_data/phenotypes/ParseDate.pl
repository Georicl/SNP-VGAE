#!/usr/bin/perl


$Seasons->{'01'} =       "winter";
$Seasons->{'02'} =       "winter";
$Seasons->{'03'} =       "spring";
$Seasons->{'04'} =       "spring";
$Seasons->{'05'} =       "spring";
$Seasons->{'06'} =       "summer";
$Seasons->{'07'} =       "summer";
$Seasons->{'08'} =       "summer";
$Seasons->{'09'} =       "autumn";
$Seasons->{'1'} =       "winter";
$Seasons->{'2'} =       "winter";
$Seasons->{'3'} =       "spring";
$Seasons->{'4'} =       "spring";
$Seasons->{'5'} =       "spring";
$Seasons->{'6'} =       "summer";
$Seasons->{'7'} =       "summer";
$Seasons->{'8'} =       "summer";
$Seasons->{'9'} =       "autumn";
$Seasons->{'10'} =      "autumn";
$Seasons->{'11'} =      "autumn";
$Seasons->{'12'} =      "winter";

print "Date\tDate.Day\tDate.Month\tSeason\tDate.Year\tDate.StudyDay\tPhase\tLunar.Progress\tFull.Moon.Extent\tLunar.Day\n";
my $phase = "";
my $lastphase = "Full.Moon";
 
my $head = <>;
my @data = split (/\t/, $head);

for ($n = 0; $n <= $#data; $n++) {
$Columns->{$data[$n]} = $n;
}

while (<>) {
chomp;
my @data = split(/\t/, $_);

my $Date = $data[$Columns->{'Date'}];
#my $phase = $data[$Columns->{'Phase'}];
my $phase = $data[$#data];
#print "$data[$Columns->{'Phase'}] $phase\n";

if ($phase =~ /\W+/) {
$lastphase = $phase;
}
else {
$phase = $lastphase;
}


my $LunarProgress = $data[$Columns->{'Lunar.Progress'}];
my $FullMoonExtent  =  $data[$Columns->{'Full.Moon.Extent'}];
my $LunarDay = $data[$Columns->{'Lunar.Day'}];
my $DateStudyDay  = $data[$Columns->{'Date.StudyDay'}];

  ($day, $month, $year) = $Date =~ /(\d+)\/(\d+)\/(\d+)/;

$day =~ s/^0//;
$month =~ s/^0//;
print "$Date\t$day\t$month\t$Seasons->{$month}\t$year\t$DateStudyDay\t$phase\t$LunarProgress\t$FullMoonExtent\t$LunarDay\n";
#print "Date\tDate.Day\tDate.Month\tDate.Year\tDate.StudyDay\tPhase\tLunar.Progress\tFull.Moon.Extent\tLunar.Day\n"
}
