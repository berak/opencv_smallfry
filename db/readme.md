## various c/c++ opencv key(name) - value(cv::Mat) datastore adapters

as various platforms don't come with a filesystem (or only an ephemeral one, think heroku)

    sqlite3
    mysql
    mongo/bson

trying to store binary data, (not text with escaped zeros, or b64)

you can even save compressed (imencoded Mat's), or a whole name-list for a face-reco in a uchar Mat

