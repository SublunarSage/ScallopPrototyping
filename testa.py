
import scallopy
import torch

# Creating a module for execution
my_sum2 = scallopy.Module(
  program="""
    type digit_1(a: i32), digit_2(b: i32)
    rel sum_2(a + b) = digit_1(a) and digit_2(b)
  """,
  input_mappings={"digit_1": range(10), "digit_2": range(10)},
  output_mappings={"sum_2": range(19)},
  provenance="difftopkproofs")

# Invoking the module with torch tensors. `result` is a tensor of 16 x 19
result = my_sum2(
  digit_1=torch.softmax(torch.randn(16, 10), dim=0),
  digit_2=torch.softmax(torch.randn(16, 10), dim=0))


print(result)

print("Hi!")