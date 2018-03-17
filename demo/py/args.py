import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-w", "--width", required=True)
args = vars(ap.parse_args())

print(args)
