from paleo.profiler import Profiler

netspec_files = ['paleo/nets/alex_v2.json']
device_name = 'TITAN_X'
network_name ='ethernet'
batch_size = 128
use_pipeline = False
use_only_gemm = True
num_workers = '1,2,4,8,16,32,64,128'
scaling = 'weak,strong'
ppp_comp = 1.0
ppp_comm = 1.0
separator = '\t'
parallel = 'model'
hybrid_workers = 1

num_workers = [int(x) for x in num_workers.split(',')]

for netspec_file in netspec_files:
    print(netspec_file)
    profiler = Profiler(netspec_file, separator=separator)
    profiler.simulate(device_name, network_name, batch_size, use_pipeline,
                      use_only_gemm, num_workers, scaling, ppp_comp,
                      ppp_comm, parallel, hybrid_workers)