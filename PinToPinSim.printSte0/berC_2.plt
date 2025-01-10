load "C:\Users\AHZ\Desktop\hspice_ami\PinToPinSim.printSte0\gp_common_map_c"
set cntrparam levels discrete -0.002,-0.03,-0.15,-0.46,-1.13,-2.33,-4.32,-7.37,-11.8,-18.0
set title "Error Probability (log10) at Port=2"
spl "C:\Users\AHZ\Desktop\hspice_ami\PinToPinSim.printSte0\data_map_2" us 1:2:4 w l lw 1
