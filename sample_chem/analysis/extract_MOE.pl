#!/usr/bin/perl -w

use strict;
use warnings;
use Getopt::Std;

our $Explanation = <<EXP;

Usage: $0 [-h]

  -h: show this message

  -f: MOE result file
  -d: directory        ("gpcr_MOE", "kinase_MOE")

EXP

getopts("hf:d:");
our($opt_h, $opt_f, $opt_d);
our $MOE_result_file = $opt_f if (defined $opt_f);
our $directory       = $opt_d if (defined $opt_d);

die "$Explanation\n" if (!(defined $opt_f && defined $opt_d));

our @TYPE_ID = qw/? D A d a Od Oa I C M R ?/;

our %REC      = ();
our %CHAIN    = ();
our %LIGCHAIN = ();
our %LIGID    = ();
our %AA       = ();
our @TITLE    = ();
open(my $IN, "<", "$MOE_result_file") or die "$!: $MOE_result_file";
while (my $line = <$IN>) {
    chomp $line;
    if ($line =~ /^mol/) {
	@TITLE = split(/,/, $line);
    } else {
	my @DATA = split(/,/, $line);
	my $pdb_id = $DATA[1];
	%REC   = ();
	%CHAIN = ();
	%LIGCHAIN = ();
	%LIGID    = ();
	&get_pdb($directory, $pdb_id);
	%AA = ();
	&get_seq($directory, $pdb_id);
	my $PLIF_raw = $DATA[11];
	$PLIF_raw =~ /\[\[([\d.\s]+)\] \[([\d.\s-]+)\] \[([\d.\s-]+)\] \[([\d.\s]+)\] \[([\d.\se-]+)\]\]/;
	my $ligidx_list   = $1; my @LIGIDX   = split(/\s+/, $ligidx_list);
	my $recidx_list   = $2; my @RECIDX   = split(/\s+/, $recidx_list);
	my $recuid_list   = $3; my @RECUID   = split(/\s+/, $recuid_list);
	my $type_list     = $4; my @TYPE     = split(/\s+/, $type_list);
	my $strength_list = $5; my @STRENGTH = split(/\s+/, $strength_list);

	my %VALUE = ();
	for my $i(0..$#LIGIDX) {
	    my $ligidx   = $LIGIDX[$i];
	    my $recidx   = $RECIDX[$i];
	    my $type     = $TYPE[$i];
	    my $strength = $STRENGTH[$i];
	    my $ligid    = $LIGID{$ligidx};
	    my $recuid   = $RECUID[$i];
#	    next if (!defined $REC{$recidx});
#	    next if ($RECIDX[$i] == 0);
	    next if (!defined $LIGCHAIN{$ligidx});
	    my $chain  = $LIGCHAIN{$ligidx};
#	    my $recuid = $REC{$recidx};
	    next if (!defined $AA{$chain."_".$recuid});
	    $VALUE{sprintf("%05d", $recidx)."_".$chain."_".$recuid."_".$ligid."_".$TYPE_ID[$type]} += $strength;
	}
	my %VALUE_CHAIN = ();
	foreach my $id(sort {$a cmp $b} keys(%VALUE)) {
	    next if ($id =~ /C$/ && $VALUE{$id} < 20);
	    my $value = $VALUE{$id};
	    my ($recidx, $chain, $recuid, $ligid, $type) = split(/\_/, $id);
	    $recidx = sprintf('%d', $recidx);
	    my $aa_gcn = $AA{$chain."_".$recuid};

#	    $VALUE_CHAIN{$chain."_".$ligid} .= $recidx."_".$chain."_".$recuid."_".$ligid."_".$type."_".$value."\n";
	    $VALUE_CHAIN{$chain."_".$ligid} .= $pdb_id."_".$chain."_".$ligid."\t".$recuid.":".$aa_gcn."_".$type."_".$value."\n";
	}

	foreach my $chain_ligid(sort {$a cmp $b} keys(%VALUE_CHAIN)) {
	    print $VALUE_CHAIN{$chain_ligid};
	}
    }
}
close($IN);

sub get_pdb {
    my ($directory, $pdb_id) = @_;

    my $pdb_file = "$directory/$pdb_id\.pdb";
    my $ligidx = 0;
    open(my $IN, "<", "$pdb_file") or die "$!: $pdb_file";
    while (my $line = <$IN>) {
	if ($line =~ /^ATOM/) {
	    my $idx    = substr($line,  6, 5);
	    my $chain  = substr($line, 21, 1);
	    my $recuid = substr($line, 23, 3);
	    $idx    =~ s/^\s+//g;
	    $recuid =~ s/^\s+//g;
	    $CHAIN{$idx} = $chain;
	    $REC{$idx}   = $recuid;
	} elsif ($line =~ /^HETATM/) {
	    $ligidx += 1;
	    my $ligid = substr($line, 17, 3);
	    my $chain = substr($line, 21, 1);
	    $LIGCHAIN{$ligidx} = $chain;
	    $LIGID{$ligidx}    = $ligid;
	}
    }
    close($IN);
}

sub get_seq {
    my ($directory, $pdb_id) = @_;

    my $Naa_file = "$directory/$pdb_id\_Naa.list";
    my $gcnid = 1;
    my $pre_chain = "Z";
    open(my $IN, "<", "$Naa_file") or die "$!: $Naa_file";
    while (my $line = <$IN>) {
	chomp $line;

	my ($chain, $recuid, $aa) = split(/[:\s+]+/, $line);

	if ($pre_chain ne "Z" && $pre_chain ne $chain) {
            $gcnid = 1;
	}

	$AA{$chain."_".$recuid} = $aa.":".$gcnid;
	++$gcnid;
	$pre_chain = $chain;
    }
    close($IN);
}

