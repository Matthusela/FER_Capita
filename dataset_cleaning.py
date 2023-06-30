import pandas as pd

# Read kaggle data set from wherever saved
df = pd.read_csv(r"C:\Users\matt8\Downloads\faceset1\icml_face_data.csv")

# Ignore usage column
df = df.drop(" Usage",axis=1).copy()

# From source above this is how the emotion codes correlate to the emotions
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

# Converting pixel data into usable form that TF (tensorflow) will like later on 
df.columns = ["emotion","pixels"]
df["pixels_list"] = df.pixels.map(lambda x: np.fromstring(x, dtype=np.uint16, sep=' '))

# Quickly defining two functions to help ensure images have been pixelated properly
def image_dimension(data_frame):
    pxls = data_frame.pixels_list.apply(len)
    if pxls.isin(np.array(pxls)).all() != True:
        return "Image pixelation is not consistent across images"
    else:
        length = np.array(pxls,dtype =np.uint16)[0]
        dim = math.sqrt(length)
        if dim%1 == 0:
            return int(dim)
        else:
            return "Image pixelation has not produced a square image"
        
def number_of_images(data_frame):
    return len(data_frame.index)

inp_dim = image_dimension(df)
num_dim = number_of_images(df)

# Image representation is by standard a sqaure grid of pixels, from this dataset each image is a 2D square matrix of size 48
# Reflecting the matrix is done through:
pixels_mat =  df.pixels_list.apply(lambda x : x.reshape(inp_dim,inp_dim))
df["pixels_mat"] = pixels_mat

df.info()
# Would now Pickle the dataframe so these steps don't have to be repeated
df.to_pickle("NAME.pkl")


