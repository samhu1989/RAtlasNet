#include <torch/torch.h>
#include <vector>

///TMP
//#include "common.h"
/// NOT TMP
	

int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2);

int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2);

int interp_cuda_forward(at::Tensor z,at::Tensor prob,at::Tensor idx,at::Tensor w,at::Tensor p);

int interp_cuda_backward(at::Tensor grad,at::Tensor idx,at::Tensor w,at::Tensor gradp);

int knn_cuda(at::Tensor xyz,at::Tensor k,at::Tensor dist,at::Tensor idx);

int chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2) {
    return chamfer_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
}

int chamfer_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, 
					  at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2) {

    return chamfer_cuda_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2);
}

int knn(at::Tensor xyz,at::Tensor k,at::Tensor dist,at::Tensor idx){
    return knn_cuda(xyz,k,dist,idx);
}

int interp_forward(at::Tensor z,at::Tensor prob,at::Tensor idx,at::Tensor w,at::Tensor p){
    return interp_cuda_forward(z,prob,idx,w,p);
}

int interp_backward(at::Tensor grad,at::Tensor idx,at::Tensor w,at::Tensor gradp){
    return interp_cuda_backward(grad,idx,w,gradp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");
  m.def("backward", &chamfer_backward, "chamfer backward (CUDA)");
  m.def("knn", &knn, "knn (CUDA)");
  m.def("interp_forward",&interp_forward,"bilinear interp (CUDA)");
  m.def("interp_backward",&interp_backward,"bilinear interp (CUDA)");
}