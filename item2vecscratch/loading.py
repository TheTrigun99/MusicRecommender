import pandas as pd

def load_playlists(path):
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip"
    )
    df.columns = df.columns.str.strip().str.strip('"')

    # groupby par playlistname → liste de tracks
    playlists = (
        df.groupby("playlistname")["trackname"]
          .apply(list)
          .tolist()
    )
    return playlists

def load500(path):
    df = pd.read_csv(
    path,
    on_bad_lines="skip",
    nrows=50000,
    engine="c")
    df.columns = df.columns.str.strip().str.strip('"')

    # groupby par playlistname → liste de tracks
    playlists = (
        df.groupby("playlistname")["trackname"]
          .apply(list)
          .tolist()
    )
    return playlists

