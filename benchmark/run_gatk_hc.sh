#!/bin/bash

# --- Configuration ---
GATK_BIN="path/to/gatk"

# The reference could be downloaded from https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/references/GRCh38/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta.gz
REF="/path/to/reference/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta.gz"

# The BAM file could be downloaded from https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/Element_AVITI_20240920/HG001_Element-LngInsert_30x_GRCh38-GIABv3.bam
BAM="/path/to/bam/HG001_Element-LngInsert_30x_GRCh38-GIABv3.bam"

# Remember to download the index files for both the reference and BAM, and place them in the same directory as the respective files!

# Clean cache function
clear_cache() {
    echo "Cleaning system cache (Drop Caches)..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "Warning: Insufficient permissions, unable to clear cache. Please ensure docker run is executed with --privileged"
    sleep 5
}

# Run and statistics function
run_hc() {
    local threads=$1
    local output=$2
    local stat_file=$3
    
    echo "--------------------------------------"
    echo "Thread = $threads"
    clear_cache
    
    /usr/bin/time -v $GATK_BIN HaplotypeCaller \
        -I $BAM -R $REF -O $output \
        --native-pair-hmm-threads $threads \
        > "${output}.log" 2>&1
        
    echo "Test completed. Results saved to ${output}"
}

# --- Execution Flow ---

# 1. Default test (DEFAULT: 4 threads)
run_hc 4 "/data/hg001/hc_single.vcf" "stats_single.txt"
# Extract key data and rename statistics file
mv /tmp/time_out_temp "stats_single.txt" 2>/dev/null 

# 2. Multi-thread test (50 threads)
run_hc 50 "/data/hg001/hc_multi.vcf" "stats_multi.txt"

echo "--------------------------------------"
echo "All tests completed!"
echo "Please check the log files for each run, or use grep 'Percent of CPU' and 'Elapsed' to view statistics."