#include "region_interface.h"
#include "../ActiveRegion/dbsnp_manager.h"

RegionResource::~RegionResource()
{
    delete pool_;
    delete[] buffer_;
    delete[] assemble_result_buffer_;
    if (db_manager) delete db_manager;
}
