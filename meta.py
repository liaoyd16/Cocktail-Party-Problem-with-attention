proj_root = "/Users/liaoyuanda/Desktop/DAE_Cocktail/"
logroot = "/Users/liaoyuanda/Desktop/DAE_Cocktail/log/"


wav_clip_root = "/Users/liaoyuanda/Desktop/DAE_Cocktail/wav_clips/"
dataset = "/Users/liaoyuanda/Desktop/DAE_Cocktail/dataset/"


training_result = "/Users/liaoyuanda/Desktop/DAE_Cocktail/training_result/"
result = "/Users/liaoyuanda/Desktop/DAE_Cocktail/result/"


def clear_dir(abs_path, kw = ""):
    import os
    os.chdir(abs_path)
    filelist = os.listdir()
    for f in filelist:
        if not kw=="" and kw not in f: continue
        os.remove(f)
    os.chdir(proj_root)