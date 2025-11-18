import os
import warnings
import urllib.request
import ssl


pretrained_sevirlr_vae_name = "pretrained_sevirlr_vae_8x8x64_v1.pt"
pretrained_sevirlr_earthformerunet_name = "pretrained_sevirlr_earthformerunet_v1.pt"
pretrained_sevirlr_alignment_name = "pretrained_sevirlr_alignment_avg_x_cuboid_v1.pt"

pretrained_i3d_400_name = "pretrained_i3d_400.pt"
pretrained_i3d_600_name = "pretrained_i3d_600.pt"

file_id_dict = {
    pretrained_sevirlr_vae_name: "ifknrw.sn.files.1drv.com/y4mHwaGuDSwdB23r1gQvSPgRbxYnqDJleZA725wzW0qHyGvDWITlvKgWzEGJeqNSH_QUpMWOsD3-DRgtT4FFR3FjbjrSCpq9mA7qMdoORSl6un65pJ0mpFLN0iukIGAPVC4AIIZNJAq8LQIo6KVZUOdjSEbBtfaKDwislTnL7lGI15VSwX2DNXBbLWjqTzy6afmBFFxVDArCrkjtbs_st4prWEqAcNF559zLQ8zhdKEv2s",
    pretrained_sevirlr_earthformerunet_name: "xiiula.sn.files.1drv.com/y4m7WESstIWGBGRDr5u4Zs-lYTF1Pf1XNOajmULFfBNcEJ5c0du5L2uy8NPDtbueTv15gIZaVQF7-YLtOaLrkD8KLjnEr_KFVGpVd-K3nt6cWyGcQZqaJvNr8wLUr6AacMh7t6PGuts2B1OKHYqzssr5HAtS3tr80sIRhgNekxV5mcxdlphj4hdghaelP01O4mdNCSKgewMO3Qn-zQDlM_7SZwSCl-fH70hbMzBfYN62wQ",
    pretrained_sevirlr_alignment_name: "ifjj0a.sn.files.1drv.com/y4m-gMKkZaeLoKWTnFenWWprSOPDh4mBkqCr7J14TmwOq8HvBcuA4gqR9I30jriOrIUsfVfcepKEWpDbIn0c5hg5wZ03LhdFp71eu5Vhj9T1dzSA0QOumdbWy2tpOLj2NOOz8HD55LWRJdvMGx6K2qWyN8MDvAJ-2256gVGbHgSzTGXBCJfe8UKm5NXxW3YWrv5-dme11fvPGqDqEQjVMQmYLQd0v6mPJBg4BJBM6n31PQ",
    pretrained_i3d_400_name: "iflkra.sn.files.1drv.com/y4mG3BFtl8R9gOYQrPsgR_E0iEGBC59OZbq5fJclm8s7Gjybz4FLFXX8K1i4AVKI0brk0OdlHL8sbPLIREyLY6PkLKCqeXwzu4GCfsyY-70WlDR7L-M3qYHBT4kAjKHVtf8WoEm2gIlCA6m6F-nMZUnEb8jUc6QL550RABAr2cLmyU1Au3kDdX4GVCPL0sM58boeiutB3kumRd6UuqabGTcOrEnLUWue8gOjGIJfuigQyc",
    pretrained_i3d_600_name: "xijdzg.sn.files.1drv.com/y4m9JgY-oUwOYkAEXDEU7j3OI-btR_048Sf9XLY-fMrDmc6xIoTCB8r76RoZE1Fq8Ha5qRFKIMSlMVuACe8rEXrfzaztxI91daUeQe4ELUua1jACAxq1Yozq0cnXXbR6ah_kjHV6cABXlzSVTt4lhjyEEEZ08ntbj-R57WrbmsXyztrudLzMKlKq7AP6O3qlx7yAS1BwH07hPd8ufKd1mb0Dp2q1nR8fgRWtKiJt9AWjLI",
}


def download_pretrained_weights(ckpt_name, save_dir=None, exist_ok=False):
    r"""
    Download pretrained weights using Python (no external wget required).

    Parameters
    ----------
    ckpt_name:  str
    save_dir:   str
    exist_ok:   bool
    """
    if save_dir is None:
        from .path import default_pretrained_dir
        save_dir = default_pretrained_dir
    ckpt_path = os.path.join(save_dir, ckpt_name)
    if os.path.exists(ckpt_path) and not exist_ok:
        warnings.warn(f"Checkpoint file {ckpt_path} already exists!")
        return
    os.makedirs(save_dir, exist_ok=True)
    file_id = file_id_dict[ckpt_name]
    url = f"https://{file_id}?AVOverride=1"
    # Some environments need an unverified SSL context for OneDrive links
    try:
        ctx = ssl.create_default_context()
        urllib.request.urlretrieve(url, ckpt_path)
    except Exception:
        try:
            ctx = ssl._create_unverified_context()
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, ckpt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download pretrained weights from {url} to {ckpt_path}: {e}")