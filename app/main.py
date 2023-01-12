from app.fma.data.fma import fma
from playground.utils.utils import disable_cuda

disable_cuda()

if __name__ == "__main__":
    # fma.make_spectograms("Electronic")
    # fma.data_init()
    fma.train_fma(epochs=90)
