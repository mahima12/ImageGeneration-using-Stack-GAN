a = "C:/Users/user/Desktop/st/8Sem/skip-thoughts-master/StackGAN-master/models/demo/stageI/stageII/misc/Data/birds/example_captions.txt"
  #  cap_path = "stageI\\stageII\\misc\\Data\\birds\\example_captions.txt"
    print(a)
    with open(a) as f:
        captions = f.read().split('\n')
        print(captions)

    captions_list = [cap for cap in captions if len(cap) > 0]
    print(captions_list)
    print('Total number of sentences:', len(captions_list))
        
        
        