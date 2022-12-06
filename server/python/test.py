from rpma_server import PMemDNNCheckpoint, CheckpointSystem

size = 16*1024*1024*1024
chksys = CheckpointSystem("/dev/dax0.0", size, False, False)
assert(chksys is not None)
ext = chksys.existing_chkpts()
assert("aaa" in ext)
model_aaa = chksys.get_chkpt("aaa")
print(model_aaa.name())