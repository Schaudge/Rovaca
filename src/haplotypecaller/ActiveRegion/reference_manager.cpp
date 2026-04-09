#include "reference_manager.h"

#include <algorithm>
#include <iostream>

#include "rovaca_logger.h"
void ReferenceManager::run()
{
    int tid = current_tid.load(std::memory_order_relaxed);
    while (tid < (int)contig.key.size() && !force_finish_) {
        std::unique_lock<std::mutex> lock(m_mutex_);
        while ((size_t)loaded_count >= prefetch_number && !force_finish_) {
            m_not_full_condition_.wait(lock);
        }
        if (force_finish_) {
            m_new_comming_condition_.notify_all();
            break;
        }
        if (!bed_loader) {
            char *raw_chr_bases = fasta_loader.fetch_target_seq(tid, 0, contig.idict[tid]);
            std::shared_ptr<char> chr_bases(raw_chr_bases, free);
            ref_vec[tid] = chr_bases;
        }
        else {
            const std::vector<std::string> &keys = bed_loader->get_bed_keys();
            const std::string &chrom = contig.key[tid];
            if (std::find(keys.begin(), keys.end(), chrom) == keys.end()) {
                // skip chroms not in bed
            }
            else {
                //
                const std::map<std::string, p_bed_intervals> &intervals = bed_loader->get_bed_intervals();
                p_bed_intervals bi = intervals.at(chrom);
                p_bed_intervals new_bi = BedLoader::init_bed_intervals();
                char *raw_chr_bases = (char *)malloc(sizeof(char) * contig.idict[tid]);

                for (int i = 0; i < bi->n; i++) {
                    if (i == 0) {
                        new_bi->start[0] = bi->start[i] - padding < 0 ? 0 : bi->start[i] - padding;
                        new_bi->end[0] =
                            bi->end[i] + padding > (hts_pos_t)contig.idict[tid] ? contig.idict[tid] : bi->end[i] + padding;
                        new_bi->n++;
                    }
                    else if (new_bi->end[new_bi->n - 1] > bi->start[i] - padding) {
                        new_bi->end[new_bi->n - 1] = bi->end[i] + padding;
                    }
                    else {
                        if (new_bi->m == new_bi->n - 1) {
                            new_bi->m = new_bi->m << 1;
                            new_bi->start = (hts_pos_t *)realloc(new_bi->start, sizeof(hts_pos_t) * new_bi->m);
                            new_bi->end = (hts_pos_t *)realloc(new_bi->end, sizeof(hts_pos_t) * new_bi->m);
                        }
                        new_bi->start[new_bi->n] = bi->start[i] - padding;
                        new_bi->end[new_bi->n] =
                            bi->end[i] + padding > (hts_pos_t)contig.idict[tid] ? contig.idict[tid] : bi->end[i] + padding;
                        ;
                        new_bi->n++;
                    }
                }
                for (int i = 0; i < new_bi->n; i++) {
                    char *refbase = fasta_loader.fetch_target_seq(tid, new_bi->start[i], new_bi->end[i]);
                    memcpy(raw_chr_bases + new_bi->start[i], refbase, (new_bi->end[i] - new_bi->start[i]) * sizeof(char));
                    free(refbase);
                }
                std::shared_ptr<char> chr_bases(raw_chr_bases, free);
                ref_vec[tid] = chr_bases;
                free(new_bi->start);
                free(new_bi->end);
                free(new_bi);
            }
        }
        loaded_count++;
        tid++;
        current_tid.store(tid, std::memory_order_release);
        m_new_comming_condition_.notify_one();
    }
    return;
}

std::shared_ptr<char> ReferenceManager::get(int tid, int &length)
{
    if (current_tid.load(std::memory_order_acquire) > tid) {
        length = contig.idict[tid];
        return ref_vec[tid];
    }
    std::unique_lock<std::mutex> lock(m_mutex_);
    while (current_tid.load(std::memory_order_relaxed) <= tid && !force_finish_) {
        m_new_comming_condition_.wait(lock);
    }
    if (force_finish_ && current_tid.load(std::memory_order_relaxed) <= tid) {
        return nullptr;
    }
    length = contig.idict[tid];
    return ref_vec[tid];
}
void ReferenceManager::pop()
{
    std::unique_lock<std::mutex> lock(m_mutex_);
    while (current_tid.load(std::memory_order_relaxed) <= last_tid && !force_finish_) {
        m_new_comming_condition_.wait(lock);
    }
    if (force_finish_) return;
    ref_vec[last_tid].reset();
    last_tid++;
    loaded_count--;
    m_not_full_condition_.notify_one();
}

void ReferenceManager::assign_finish()
{
    std::unique_lock<std::mutex> lock(m_mutex_);
    force_finish_ = true;
    m_not_full_condition_.notify_one();
    m_new_comming_condition_.notify_all();
}
