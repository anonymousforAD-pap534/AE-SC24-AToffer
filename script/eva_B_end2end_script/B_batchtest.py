import os
import csv
import sys

platform = int(sys.argv[1])

for mask in [1, 2]:
    maksname = ''
    
    if mask == 1:
        maskname = "Strided"
    else:
        maskname = "Fixed"
        
    filename = "{}_end2end_{}_test.csv".format(platform, maskname)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_size'] + [str(s) for s in [64, 128, 256, 384, 512, 768, 1024]])
        
        for batch_size in [1, 8, 16]:
            batch_results = []
            read_data = []
            
            for seqlen in [64, 128, 256, 384, 512, 768, 1024]:
                command = "python {}{}_mask_fused_forward.py {} 12 {} 12 64 {}".format(seqlen, maskname, batch_size, seqlen, mask)
                output = os.popen(command).read()
                time_value = float(output.split("time costs:")[1].split("ms")[0].strip())
                read_data.append(time_value)
            writer.writerow([str(batch_size)] + [str(t) for t in read_data])
                
            
            read_data.clear()
            for seqlen in [64, 128, 256, 384, 512, 768, 1024]:
                command = "python B1_jit_all_fused.py {} 12 {} 12 64 {}".format(batch_size, seqlen, mask)
                output = os.popen(command).read()
                time_value = float(output.split("time costs:")[1].split("ms")[0].strip())
                read_data.append(time_value)
            writer.writerow([str(batch_size)] + [str(t) for t in read_data])
                
            
            read_data.clear()  
            for seqlen in [64, 128, 256, 384, 512, 768, 1024]:
                command = "python B1_no_fused.py {} 12 {} 12 64 {}".format(batch_size, seqlen, mask)
                output = os.popen(command).read()
                time_value = float(output.split("time costs:")[1].split("ms")[0].strip())
                read_data.append(time_value)
            writer.writerow([str(batch_size)] + [str(t) for t in read_data])
                
    file.close()
