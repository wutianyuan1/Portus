from rpma_server import PMemDNNCheckpoint, CheckpointSystem

size = 16*1024*1024*1024
chksys = CheckpointSystem("/dev/dax0.0", size, False, False)
assert(chksys is not None)
ext = chksys.existing_chkpts()
assert("aaa" in ext)
model_aaa = chksys.get_chkpt("aaa")
layer_info = model_aaa.get_layers_info()
print(layer_info)
for i in range(3):
    assert(f'module.layer{i+1}' == layer_info[i][0])
    assert(layer_info[i][1] == 419430400)
print("Test Passed!")
