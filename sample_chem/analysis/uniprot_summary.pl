#!/usr/bin/perl -w

use strict;
use warnings;
use Getopt::Std;

our $Explanation = <<EXP;

Usage: $0 [-h]

  -h: show this message

  -m: MOE_result/gpcr_MOE_result.txt
  -g: GCN_result/gpcr_GCN_max488_result.txt
  -f: GCN_result/gpcr_pdb_id_list.txt
  -t: active or all
  -i: gpcr_max488, gpcr_max1212

EXP

getopts("hm:g:f:t:i:");
our($opt_h, $opt_m, $opt_g, $opt_f, $opt_t, $opt_i);
our $MOE_result_file = $opt_m if (defined $opt_m);
our $GCN_result_file = $opt_g if (defined $opt_g);
our $pdbid_file      = $opt_f if (defined $opt_f);
our $decision        = $opt_t if (defined $opt_t);
our $output_index    = $opt_i if (defined $opt_i);

die "$Explanation\n" if (!(defined $opt_m && defined $opt_g && defined $opt_f && defined $opt_t  && defined $opt_i));

#
# get MOE result
#
our %MOE_RESULT = ();
our %MOE_TYPE   = ();
our %PDBID_CHAIN = ();
&get_MOE_result($MOE_result_file);

#
# get GCN result
#
our %GCN_RESULT = ();
&get_GCN_result($GCN_result_file, $pdbid_file, $decision);


our @TOTAL_IG       = ();
our @PLIF_IG        = ();
our @nonPLIF_IG     = ();
our @PLIF_TYPE      = ();
our @TOTAL_IG_ave   = ();
our @PLIF_IG_ave    = ();
our @nonPLIF_IG_ave = ();
foreach my $pdbid_chain(keys(%PDBID_CHAIN)) {
    my $TOTAL_IG   = my $TOTAL_num   = 0;
    my $PLIF_IG    = my $PLIF_num    = 0;
    my $nonPLIF_IG = my $nonPLIF_num = 0;
    foreach my $GCN_num(keys(%{$GCN_RESULT{$pdbid_chain}})) {
	my $IG = $GCN_RESULT{$pdbid_chain}{$GCN_num};
	if (defined $MOE_RESULT{$pdbid_chain}{$GCN_num}) {
	    my $type = $MOE_TYPE{$pdbid_chain}{$GCN_num};
	    $PLIF_IG += $IG;
	    push(@PLIF_IG, $IG);
	    push(@PLIF_TYPE, $type);
	    ++$PLIF_num;
	} else {
	    $nonPLIF_IG += $IG;
	    push(@nonPLIF_IG, $IG);
	    ++$nonPLIF_num;
	}
	$TOTAL_IG += $IG;
	push(@TOTAL_IG, $IG);
	++$TOTAL_num;
    }
    print STDERR $pdbid_chain."\n" if ($PLIF_num == 0);
    next if ($PLIF_num == 0);


    push(@TOTAL_IG_ave,   $TOTAL_IG   / $TOTAL_num  );
    push(@PLIF_IG_ave,    $PLIF_IG    / $PLIF_num   );
    push(@nonPLIF_IG_ave, $nonPLIF_IG / $nonPLIF_num);
}

our $output_file_ave = "histogram/result_average_".$output_index.".txt";
open(my $AVE, ">", "$output_file_ave") or die "$!: $output_file_ave";
print $AVE "total_ave_$output_index,PLIF_ave_$output_index,nonPLIF_ave_$output_index\n";
for my $i(0..$#TOTAL_IG_ave) {
    my $total_ig_ave   = ($i <= $#TOTAL_IG_ave)? $TOTAL_IG_ave[$i]: "";
    my $plif_ig_ave    = ($i <= $#PLIF_IG_ave)?  $PLIF_IG_ave[$i]: "";
    my $nonplif_ig_ave = ($i <= $#nonPLIF_IG_ave)? $nonPLIF_IG_ave[$i]: "";
    print $AVE $total_ig_ave.",".$plif_ig_ave.",".$nonplif_ig_ave."\n";
}
close($AVE);

our $output_total_file = "histogram/result_".$output_index."_total.txt";
open(my $TOTAL, ">", "$output_total_file") or die "$!: $output_total_file";
print $TOTAL "total_$output_index\n";
for my $i(0..$#TOTAL_IG) {
    my $total_ig = $TOTAL_IG[$i];
    print $TOTAL $total_ig."\n";
}
close($TOTAL);
our $output_PLIF_file = "histogram/result_".$output_index."_PLIF.txt";
open(my $PLIF, ">", "$output_PLIF_file") or die "$!: $output_PLIF_file";
print $PLIF "PLIF_$output_index,type\n";
for my $i(0..$#PLIF_IG) {
    my $plif_ig = $PLIF_IG[$i];
    my $plif_type = $PLIF_TYPE[$i];
    print $PLIF $plif_ig.",".$plif_type."\n";
}
close($PLIF);
our $output_nonPLIF_file = "histogram/result_".$output_index."_nonPLIF.txt";
open(my $NONPLIF, ">", "$output_nonPLIF_file") or die "$!: $output_nonPLIF_file";
print $NONPLIF "nonPLIF_$output_index\n";
for my $i(0..$#nonPLIF_IG) {
    my $nonplif_ig = $nonPLIF_IG[$i];
    print $NONPLIF $nonplif_ig."\n";
}
close($NONPLIF);


sub get_MOE_result {
    my ($MOE_result_file) = @_;

    open(my $IN, "<", "$MOE_result_file") or die "$!: $MOE_result_file";
    while (my $line = <$IN>) {
	chomp $line;
	
	my ($pair, $data) = split(/\t/, $line);
	my $pdbid_chain = substr($pair, 0, 6);
	my ($MOE_num, $aa, $GCN_num, $type, $value) = split(/[:_]+/, $data);
	if ($GCN_result_file =~ /uniprot/) {
	    $MOE_RESULT{$pdbid_chain}{$GCN_num} = $value;
	    $MOE_TYPE{$pdbid_chain}{$GCN_num}   = $pair."_".$data;
	} else {
	    $MOE_RESULT{$pdbid_chain}{$GCN_num} = $value;
	    $MOE_TYPE{$pdbid_chain}{$GCN_num}   = $pair."_".$data;
	}
    }
}

sub get_GCN_result {
    my ($GCN_result_file, $pdbid_file, $decision) = @_;

    my %PDB = ();
    open(my $ID, "<", "$pdbid_file") or die "$!: $pdbid_file";
    while (my $line = <$ID>) {
	chomp $line;
	my ($id, $pdbid_chain) = split(/\t/, $line);
	$PDB{$id} = $pdbid_chain;
	$PDBID_CHAIN{$pdbid_chain} = 1;
    }
    close($ID);

    my $pdbid_chain = "";
    my $GCN_num = 1;
    my $active_or_inactive = "";
    open(my $IN, "<", "$GCN_result_file") or die "$!: $GCN_result_file";
    while (my $line = <$IN>) {
	chomp $line;
	if ($line =~ /^mol/) {
	    my $jbl_file = $line;
	    $jbl_file =~ /mol_(\d\d\d\d)_task_0_(\w+)_all/;
	    my $id = $1;
	    $active_or_inactive = $2;
	    $id = sprintf('%d', $id);
	    $pdbid_chain = $PDB{$id};
	    $GCN_num = 1;
	} else {
	    if ($decision eq "all" || $active_or_inactive eq $decision) {
		my ($aa, $IG) = split(/\t/, $line);
		next if ($IG == 0.0);
		$GCN_RESULT{$pdbid_chain}{$GCN_num} = $IG;
		++$GCN_num;
	    }
	}
    }
    close($IN);
}
