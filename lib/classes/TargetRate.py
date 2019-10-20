
##############################################################
# This is the previous code. Edit and finalize for version-3 #
##############################################################

import numpy as np

class TargetRate:    
    def __init__(self, length=0, offset=0, amplitude=1, mesa_top=8, gain=3, rate_type="sigmoid"):
        '''
        __param__ length: length of the generated series
        __param__ offset: tuple. first is the initial offset, second bool is whether or not offset is variable
        __param__ amplitude: tuple. first is the initial amplitude, second bool is whether or not amplitude is variable
        '''
        self.length = length
        self.offset = offset
        self.amplitude = amplitude
        self.mesa_top = mesa_top
        self.gain = gain
        self.rate_type = rate_type
        
    def generate_series(self, n_filters):
        '''
        generate_series: actually generates the series this target rate object represents
        
        __return__ target_rate_series: series represented by target rate object
        '''
        
        '''if self.length == 0:
                                    print("Error: length of target series cannot equal zero. Set length of target_shape object")
                                    return -1
                                
                                beg = 0
                                end = int(self.length)
                                
                                target_rate_series_list = []
                                for i in range(n_filters):
                                    target_rate_series = np.linspace(beg,end,num=self.length)
                        
                                    if self.rate_type == "sigmoid":
                                        target_rate_series = self.sigmoid(x=target_rate_series, offset=self.offset, amplitude=self.amplitude)
                                    elif self.rate_type == "mesa":
                                        target_rate_series = self.mesa(x=target_rate_series, actual_offset=self.offset, amplitude=self.amplitude,
                                                                     mesa_top=self.mesa_top, gain=self.gain)
                                    elif self.rate_type == "reverse-sigmoid":
                                        target_rate_series = self.reverse_sigmoid(x=target_rate_series, offset=self.offset, amplitude=self.amplitude)
                                    else:
                                        print("Error in target rate: unrecognized rate type")
                                        break;
                                        
                                    target_rate_series_list.append(target_rate_series)
                                    
                                target_rate_series_final = np.asarray(target_rate_series_list)
                                target_rate_series_final = np.expand_dims(target_rate_series_final, axis=2)
                                return target_rate_series_final'''

        if self.length == 0:
            print("Error: length of target series cannot equal zero. Set length of target_shape object")
            return -1
        
        beg = 0
        end = int(self.length)
        
        target_rate_series = np.linspace(beg,end,num=self.length)

        if self.rate_type == "sigmoid":
            target_rate_series = self.sigmoid(x=target_rate_series, offset=self.offset, amplitude=self.amplitude)
        elif self.rate_type == "mesa":
            target_rate_series = self.mesa(x=target_rate_series, actual_offset=self.offset, amplitude=self.amplitude,
                                         mesa_top=self.mesa_top, gain=self.gain)
        elif self.rate_type == "reverse-sigmoid":
            target_rate_series = self.reverse_sigmoid(x=target_rate_series, offset=self.offset, amplitude=self.amplitude)
        else:
            print("Error in target rate: unrecognized rate type")
            return
            
        target_rate_series = np.expand_dims(target_rate_series, axis=1)
        target_rate_series = np.tile(target_rate_series, (1, n_filters))
        target_rate_series = np.expand_dims(target_rate_series, axis=0)
        return target_rate_series

    def sigmoid(self, x, offset, amplitude, gain=1):
        y = 1 / (1 + np.exp(-1*gain*(x-offset)))
        y = amplitude * y
        return y

    def mesa(self, x, actual_offset, amplitude, mesa_top, gain=1):
        offset1 = -1 * (mesa_top/2) + actual_offset
        offset2 = (mesa_top/2) + actual_offset
        
        print(offset1)
        print(offset2)
        
        sig1 = sigmoid(x, offset1, amplitude, gain=gain)
        sig2 = -1*sigmoid(x, offset2, amplitude, gain=gain)+1
        mesa = np.multiply(sig1, sig2)
        
        return mesa

    def reverse_sigmoid(self, x, offset, amplitude, gain=1):
        y = 1 / (1 + np.exp(gain*(x-offset)))
        y = amplitude * y
        
        return y
    
    def print(self):
        print("Target shape:", self.name)
        print("\tOffset:", self.offset)
        print("\tAmplitude:", self.amplitude)