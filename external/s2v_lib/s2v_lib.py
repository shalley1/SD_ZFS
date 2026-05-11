# This file is a modified version of the code found at https://github.com/Hanjun-Dai/pytorch_structure2vec

import ctypes
import numpy as np
import os
import sys
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from utils.debug import get_logger
log = get_logger("s2v_lib")

class _s2v_lib(object):

    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #import c++ symbols9=(GetGraphStruct,PrepareBatchGraph etc. from shared object compiled from Makefile)
        self.lib = ctypes.CDLL('%s/build/dll/libs2v.so' % dir_path)
        
        #set return types
        self.lib.GetGraphStruct.restype = ctypes.c_void_p
        self.lib.PrepareBatchGraph.restype = ctypes.c_int
        self.lib.PrepareMeanField.restype = ctypes.c_int
        self.lib.PrepareLoopyBP.restype = ctypes.c_int
        self.lib.NumEdgePairs.restype = ctypes.c_int
        #if using python 3, we encode str to bytes
        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args
        
        #example: python train.py -msg_average 1
        #         --> sys.argv == ["train.py", "-msg_average", "1"]
        # configuration allows for message passing to sum incoming message when msg_average=0 or message passing to average incoming messages when msg_average==1
        
        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)
        
        #python object that holds a pointer to a C++ GraphStruct instance.
        self.batch_graph_handle = ctypes.c_void_p(self.lib.GetGraphStruct())

    def _prepare_graph(self, graph_list, is_directed=0):    
        log.info("[_prepare_graph] start: num_graphs=%d is_directed=%d",
             len(graph_list), is_directed)
        
        edgepair_list = (ctypes.c_void_p * len(graph_list))()
        list_num_nodes = np.zeros((len(graph_list), ), dtype=np.int32)
        list_num_edges = np.zeros((len(graph_list), ), dtype=np.int32)        
        for i in range(len(graph_list)):
            if type(graph_list[i].edge_pairs) is ctypes.c_void_p:
                edgepair_list[i] = graph_list[i].edge_pairs
            elif type(graph_list[i].edge_pairs) is np.ndarray:
                edgepair_list[i] = ctypes.c_void_p(graph_list[i].edge_pairs.ctypes.data)
            else:
                raise NotImplementedError

            list_num_nodes[i] = graph_list[i].num_nodes
            list_num_edges[i] = graph_list[i].num_edges
            gp = graph_list[i]
            log.info("[_prepare_graph] graph[%d]: N=%d E=%d edge_pairs dtype=%s shape=%s contiguous=%s",
                     i,
                     gp.num_nodes,
                     gp.num_edges,
                     getattr(gp.edge_pairs, "dtype", type(gp.edge_pairs)),
                     getattr(gp.edge_pairs, "shape", None),
                     gp.edge_pairs.flags["C_CONTIGUOUS"]
                     if hasattr(gp.edge_pairs, "flags") else "n/a")
        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)
        log.info("[_prepare_graph] totals: total_N=%d total_E=%d",
                 total_num_nodes, total_num_edges)
        
        log.info("[_prepare_graph] calling C++ PrepareBatchGraph(...)")
        self.lib.PrepareBatchGraph(self.batch_graph_handle, 
                                len(graph_list), 
                                ctypes.c_void_p(list_num_nodes.ctypes.data),
                                ctypes.c_void_p(list_num_edges.ctypes.data),
                                ctypes.cast(edgepair_list, ctypes.c_void_p),
                                is_directed)
        
        log.info("[_prepare_graph] returned from C++ PrepareBatchGraph")
        return total_num_nodes, total_num_edges

    def PrepareMeanField(self, graph_list, is_directed=0):
        log.info("[PrepareMeanField] start")
        assert not is_directed
        total_num_nodes, total_num_edges = self._prepare_graph(graph_list, is_directed)

        log.info("[PrepareMeanField] after batching: N=%d E=%d",
                 total_num_nodes, total_num_edges)

        n2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        n2n_vals = torch.FloatTensor(total_num_edges * 2)

        e2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        e2n_vals = torch.FloatTensor(total_num_edges * 2)

        subg_idxes = torch.LongTensor(2, total_num_nodes)
        subg_vals = torch.FloatTensor(total_num_nodes)

        idx_list = (ctypes.c_void_p * 3)()
        idx_list[0] = n2n_idxes.numpy().ctypes.data
        idx_list[1] = e2n_idxes.numpy().ctypes.data
        idx_list[2] = subg_idxes.numpy().ctypes.data

        val_list = (ctypes.c_void_p * 3)()
        val_list[0] = n2n_vals.numpy().ctypes.data
        val_list[1] = e2n_vals.numpy().ctypes.data
        val_list[2] = subg_vals.numpy().ctypes.data
        
        log.info("[PrepareMeanField] calling C++ PrepareMeanField(...)")
        self.lib.PrepareMeanField(self.batch_graph_handle,
                                ctypes.cast(idx_list, ctypes.c_void_p),
                                ctypes.cast(val_list, ctypes.c_void_p))
        log.info("[PrepareMeanField] returned from C++ PrepareMeanField")
        
        #instead of using dense matrices, store only nonzero ones with spare tensors(COO- coordinate format)

        n2n_sp = torch.sparse_coo_tensor(n2n_idxes, n2n_vals, torch.Size([total_num_nodes, total_num_nodes]))
        e2n_sp = torch.sparse_coo_tensor(e2n_idxes, e2n_vals, torch.Size([total_num_nodes, total_num_edges * 2]))
        subg_sp = torch.sparse_coo_tensor(subg_idxes, subg_vals, torch.Size([len(graph_list), total_num_nodes]))
        log.info("[PrepareMeanField] built sparse tensors: n2n=%s nnz=%d | e2n=%s nnz=%d | subg=%s nnz=%d",
             tuple(n2n_sp.shape), n2n_sp._nnz(),
             tuple(e2n_sp.shape), e2n_sp._nnz(),
             tuple(subg_sp.shape), subg_sp._nnz())
        return n2n_sp, e2n_sp, subg_sp

    def PrepareLoopyBP(self, graph_list, is_directed=0):
        assert not is_directed
        total_num_nodes, total_num_edges = self._prepare_graph(graph_list, is_directed)
        total_edge_pairs = self.lib.NumEdgePairs(self.batch_graph_handle)

        n2e_idxes = torch.LongTensor(2, total_num_edges * 2)
        n2e_vals = torch.FloatTensor(total_num_edges * 2)

        e2e_idxes = torch.LongTensor(2, total_edge_pairs)
        e2e_vals = torch.FloatTensor(total_edge_pairs)

        e2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        e2n_vals = torch.FloatTensor(total_num_edges * 2)

        subg_idxes = torch.LongTensor(2, total_num_nodes)
        subg_vals = torch.FloatTensor(total_num_nodes)

        idx_list = (ctypes.c_void_p * 4)()
        idx_list[0] = ctypes.c_void_p(n2e_idxes.numpy().ctypes.data)
        idx_list[1] = ctypes.c_void_p(e2e_idxes.numpy().ctypes.data)
        idx_list[2] = ctypes.c_void_p(e2n_idxes.numpy().ctypes.data)
        idx_list[3] = ctypes.c_void_p(subg_idxes.numpy().ctypes.data)

        val_list = (ctypes.c_void_p * 4)()
        val_list[0] = ctypes.c_void_p(n2e_vals.numpy().ctypes.data)
        val_list[1] = ctypes.c_void_p(e2e_vals.numpy().ctypes.data)
        val_list[2] = ctypes.c_void_p(e2n_vals.numpy().ctypes.data)
        val_list[3] = ctypes.c_void_p(subg_vals.numpy().ctypes.data)

        self.lib.PrepareLoopyBP(self.batch_graph_handle,
                                ctypes.cast(idx_list, ctypes.c_void_p),
                                ctypes.cast(val_list, ctypes.c_void_p))

        n2e_sp = torch.sparse_coo_tensor(n2e_idxes, n2e_vals, torch.Size([total_num_edges * 2, total_num_nodes]))
        e2e_sp = torch.sparse_coo_tensor(e2e_idxes, e2e_vals, torch.Size([total_num_edges * 2, total_num_edges * 2]))
        e2n_sp = torch.sparse_coo_tensor(e2n_idxes, e2n_vals, torch.Size([total_num_nodes, total_num_edges * 2]))
        subg_sp = torch.sparse_coo_tensor(subg_idxes, subg_vals, torch.Size([len(graph_list), total_num_nodes]))

        return n2e_sp, e2e_sp, e2n_sp, subg_sp

dll_path = '%s/build/dll/libs2v.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    S2VLIB = _s2v_lib(sys.argv)
else:
    S2VLIB = None

