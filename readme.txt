First of all, pictures taken by scan where split and saved in the corresponding folders
using file split and split1.
Afterwards the pictures were resized and scaled from RGBA to RGB then to Grey and saved.
Then, the pictures were read and saved to arrays x.npy and y.npy in file data_obtaining.
These arrays were read and all other stuff like creating model split of data were done in
file model.
Actually We could not turn on the GPU therefore the code was tun in google colab.
However you need to upload the x.npy and y.npy files there.
The colab files are in colab folder. Also see the model_from_colab.
All pictures in DATA_SET folder.