
class AMI_Mask:
    def __init__(self,ddr_type = "lpddr5",clock = "3000"):

        self.ddr_type = ddr_type.lower()
        self.clock = clock

        self.mask_dict = {
            "lpddr5":{
                "tdivw1":0.35,
                "tdivw2":0.18,
                "vdivw":{
                    "1867":100,
                    "2134":100,
                    "2400":100,
                    "2750":100,
                    "3000":100,
                    "3200":100,
                    "3750":80,
                    "4266":80,
                }
            }
        }
    
    def get_mask(self,ui=False):
        tdivw1 = self.mask_dict[self.ddr_type]["tdivw1"]
        tdivw2 = self.mask_dict[self.ddr_type]["tdivw2"]
        vdivw = self.mask_dict[self.ddr_type]["vdivw"][self.clock]
        if ui:
            return tdivw1,tdivw2,vdivw
        else:
            ui_valve = 1/(float(self.clock) * 1e6)
            return tdivw1*ui_valve,tdivw2*ui_valve,vdivw*1e-3
