import os

for mask in [1, 2]:
    if(mask == 1): 
        mask_name = 'Strided_mask'
    else:
        mask_name = 'Fixed_mask'
    print("Stage{} {} train dataset generate start".format(mask, mask_name))
    
    for seqlen in [64, 128, 256, 512]:
        print("seqlen = ", seqlen, "-"*30 )
        for filename in ["search1_random", "search2_simulated_an", "search3_ours"]:
            print("{}   {}.py {} script start !".format(mask_name, filename, seqlen))
            os.system("python {}.py {} {}".format(filename, seqlen, mask))

print("Congratulations! Search finished!")
