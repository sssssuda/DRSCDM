import glob, numpy, tqdm

trackroot_path = "D:/FYH/datas/bbt_buffy/tracks"
idroot_path = "D:/FYH/datas/bbt_buffy/features"
nameroot_path = "D:/FYH/datas/bbt_buffy/ids"
saveroot_path = "D:/FYH/datas/bbt_buffy"


def get_episode_spks_ids(root_path):
    label_fname = f'{root_path}.ids'
    with open(label_fname, 'r') as fid:
        fid.readline()  # ignore header
        data = fid.readlines()  # track to name
        data = [line.strip().split() for line in data if line.strip()]
        # trackid --> name mapping
        ids = {int(line[0]): line[1] for line in data}

    # get unique names and assign numbers
    uniq_names = list(set(ids.values()))
    return ids, uniq_names


def get_episode_spks_embeds(root_path):
    id2embeds = {}
    id_files = glob.glob(f"{root_path}/*.npy")
    for filename in id_files:
        filename = filename.replace("\\", "/")
        id = int(filename.split("/")[-1].split(".")[0])
        embeds = numpy.load(filename)
        id2embeds[id] = embeds

    return id2embeds


def get_episode_stream_embeds(filename):
    npy_path = idroot_path + f"/{filename}"
    id2embeds = get_episode_spks_embeds(npy_path)
    name_path = nameroot_path + f"/{filename}"
    ids, names = get_episode_spks_ids(name_path)
    tracks_path = trackroot_path + f"/{filename}.tracks"
    stream_embeds = numpy.zeros((1, 256))
    labels = []
    with open(tracks_path, "r") as f:
        i = 2
        for line in tqdm.tqdm(f.readlines(), desc="Readlines"):
            if i > 0:
                i -= 1
                continue

            detail = line.strip().split(" ")
            for i in range(int(detail[2])):
                spk_id = int(detail[5 * i + 3])
                embed = id2embeds[spk_id][0]
                embed = numpy.reshape(embed, (1, 256))
                id2embeds[spk_id] = numpy.delete(id2embeds[spk_id], 0, axis=0)
                stream_embeds = numpy.concatenate((stream_embeds, embed), axis=0)
                labels.append(names.index(ids[spk_id]))

    stream_embeds = numpy.delete(stream_embeds, 0, axis=0)
    labels = numpy.array(labels)
    numpy.save(f"{saveroot_path}/stream_embeds/{filename}.npy", stream_embeds)
    numpy.save(f"{saveroot_path}/labels/{filename}.npy", labels)
    print(f"embeds: {stream_embeds.shape}, labels: {labels.size}")

    return stream_embeds, labels
