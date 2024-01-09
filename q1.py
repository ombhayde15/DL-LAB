import torch

tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])

reshaped_tensor = tensor_a.view(3, 2)
print("Reshaped Tensor:")
print(reshaped_tensor)

viewed_tensor = tensor_a.view(3, 2)
print("Viewed Tensor:")
print(viewed_tensor)

tensor_b = torch.tensor([[7, 8, 9], [10, 11, 12]])
stacked_tensor = torch.stack([tensor_a, tensor_b])
print("Stacked Tensor:")
print(stacked_tensor)

tensor_c = torch.tensor([[[1, 2, 3]]])
squeezed_tensor = torch.squeeze(tensor_c)
print("Squeezed Tensor:")
print(squeezed_tensor)

tensor_d = torch.tensor([1, 2, 3])
unsqueezed_tensor = torch.unsqueeze(tensor_d, dim=0)
print("Unsqueezed Tensor:")
print(unsqueezed_tensor)

# torch.permute
tensor_a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Original Tensor:")
print(tensor_a)

permuted_tensor = tensor_a.permute(2, 0, 1)
print("\nPermuted Tensor:")
print(permuted_tensor)

# Matrix Mul

tensor_e = torch.randn(7,7)
print("tensor_e:")
print(tensor_e)

random_tensor = torch.randn(1, 7)
print("\nRandom Tensor:")
print(random_tensor)

result_tensor = torch.matmul(tensor_e, random_tensor.t())
print("\nResult Tensor:")
print(result_tensor)

# GPU
tensor1 = torch.randn(2,3)
tensor2 = torch.randn(2,3)


print(tensor1,tensor1.device)
print(tensor2,tensor2.device)

tensor1_on_gpu = tensor1.to("cuda")
print(tensor1_on_gpu)
tensor2_on_gpu = tensor2.to("cuda")
print(tensor2_on_gpu)

max_value_tensor1 = torch.max(tensor1_on_gpu)
min_value_tensor1 = torch.min(tensor1_on_gpu)
max_value_tensor2 = torch.max(tensor2_on_gpu)
min_value_tensor2 = torch.min(tensor2_on_gpu)

print("Max",max_value_tensor1)
print("Min",min_value_tensor1)
print("Max",max_value_tensor2)
print("Min",min_value_tensor2)
