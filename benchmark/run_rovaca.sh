#!/bin/bash

# --- Configuration ---
ROVACA_BIN="path/to/rovaca"

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
run_rovaca() {
    local threads=$1
    local output=$2
    local stat_file=$3
    
    echo "--------------------------------------"
    echo "Thread = $threads"
    clear_cache
    
    /usr/bin/time -v $ROVACA_BIN \
        -I $BAM -R $REF -O $output \
        --nthreads $threads \
        > "${output}.log" 2>&1
        
    echo "Test completed. Results saved to ${output}"
}

# --- Execution Flow ---

#  Multi-thread test (52 threads)
run_rovaca 52 "/data/hg001/rovaca.vcf" "stats_rovaca.txt"

echo "--------------------------------------"
echo "All Rovaca tests completed!"
echo "Please check the log files for each run, or use grep 'Percent of CPU' and 'Elapsed' to view statistics."