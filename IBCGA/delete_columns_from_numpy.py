from imghdr import what
import numpy as np
def delete_columns_from_numpy_array(arr,what_to_delete):
    print(what_to_delete)
    arr = np.delete(arr,what_to_delete,1)
    return arr