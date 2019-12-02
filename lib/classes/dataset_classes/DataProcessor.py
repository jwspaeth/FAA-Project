import pickle
import json


def data_processor(file_path):
    unpickled = pickle.load(
        open(file_path, 'rb'))
    with open('data.csv', 'w') as outfile:
        data = json.dumps(unpickled)
        outfile.write(data)
    data_length = len(unpickled['data'])

    # data should go like this: [[[x,y,z],[x,y,z]],[[x,y,z],[x,y,z]], .....]
    # for the other 6 in a segment strip
    master_array = []
    # counter for segment
    step = 0

    # There are MANY rows so watch out. Make a loop. Also you can be more efficient, just illustrating here.
    for rowi in range(0, data_length):
        # Getting length of time array to know how many data points there are in total
        # across x, y, and z
        set_length = len(unpickled['data'][rowi][10])

        set_length = set_length - (set_length % 8)
        # Append sub array
        master_array.append([[[], []]])

        for data_set in range(0, set_length):
            if (data_set % 7 == 0):
                x = unpickled['data'][rowi][11][data_set]
                y = unpickled['data'][rowi][12][data_set]
                z = unpickled['data'][rowi][13][data_set]
                master_array[rowi][0][1].append([x, y, z])
            elif (data_set % 7 == 1):
                if data_set != 1:
                    x = unpickled['data'][rowi][11][data_set]
                    y = unpickled['data'][rowi][12][data_set]
                    z = unpickled['data'][rowi][13][data_set]
                    master_array[rowi][0][1].append([x, y, z])
                else:
                    x = unpickled['data'][rowi][11][data_set]
                    y = unpickled['data'][rowi][12][data_set]
                    z = unpickled['data'][rowi][13][data_set]
                    master_array[rowi][0][0].append([x, y, z])
            else:
                x = unpickled['data'][rowi][11][data_set]
                y = unpickled['data'][rowi][12][data_set]
                z = unpickled['data'][rowi][13][data_set]
                master_array[rowi][0][0].append([x, y, z])

    return master_array
