import os, glob
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
BLOCK = 20
RESULSTS_DIR = os.path.join(os.path.dirname(__file__), "results")
paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
assert paths, f"No .jpg files found in {IMAGES_DIR}"

all_blocks_flat = []
index = []
skipped_small = 0
skipped_partial = 0

for p in paths:
    try:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("L")

        arr = np.asarray(img, dtype=np.float32) / 255.0
        H, W = arr.shape

        H2 = (H // BLOCK) * BLOCK
        W2 = (W // BLOCK) * BLOCK
        if H2 < BLOCK or W2 < BLOCK:
            skipped_small += 1
            continue
        arr = arr[:H2, :W2]

        for y in range(0, H2, BLOCK):
            for x in range(0, W2, BLOCK):
                blk = arr[y:y+BLOCK, x:x+BLOCK]
                if blk.shape != (BLOCK, BLOCK):
                    skipped_partial += 1
                    continue
                all_blocks_flat.append(blk.ravel())
                index.append((os.path.basename(p), y, x))
    except Exception as e:
        print(f"Skipped {p}: {e}")

X = np.stack(all_blocks_flat, axis=0).astype(np.float32)
print(f"Images: {len(paths)} | Patches: {X.shape[0]} | Dim: {X.shape}")
print(f"Skipped small images: {skipped_small} | Skipped partial blocks: {skipped_partial}")

covar = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covar)

# np.save(os.path.join(RESULSTS_DIR, "patches_8x8.npy"), X)
# np.save(os.path.join(RESULSTS_DIR, "patches_8x8_index.npy"), np.array(index, dtype=object))

mu = X.mean(axis=0, keepdims=True)
Xc = X - X.mean(axis=0, keepdims=True)

# Sort eigenpairs
idx = np.argsort(eigenvalues)[::-1]
eigvals = eigenvalues[idx]
W = eigenvectors[:, idx]
EVR = eigvals / (eigvals.sum() + 1e-12)

# Plot first k PC filters
def plot_pc_filters(W, block=BLOCK, k=32, out_name=f"pc_filters_top32_{BLOCK}.png"):
    k = min(k, W.shape[1])
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    plt.figure(figsize=(1.8*cols, 1.8*rows))
    for i in range(k):
        f = W[:, i].reshape(block, block)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        plt.subplot(rows, cols, i+1)
        plt.imshow(f, cmap="gray")
        plt.axis("off")
        plt.title(f"PC{i+1}", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULSTS_DIR, out_name), dpi=200)
    plt.close()

# Scree plot (EVR and cumulative EVR)
def plot_evr(EVR, out_name=f"evr_scree_block{BLOCK}.png"):
    csum = np.cumsum(EVR)
    xs = np.arange(1, len(EVR)+1)
    plt.figure(figsize=(6,3.3))
    plt.plot(xs, EVR, marker="o", label="EVR")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    ax2 = plt.gca().twinx()
    ax2.plot(xs, csum, marker=".", label="Cumulative EVR")
    ax2.set_ylabel("Cumulative EVR")
    plt.title(f"Cumulative @ k=32 ≈ {csum[min(31, len(csum)-1)]:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULSTS_DIR, out_name), dpi=200)
    plt.close()

# Activation map of one PC over one image
def activation_map_for(filename, pc_idx=0, block=BLOCK, out_prefix=f"activation"):
    rows = [i for i, (fn, y, x) in enumerate(index) if fn == filename]
    assert rows, f"No patches found for {filename}"
    ys = [index[i][1] for i in rows]
    xs = [index[i][2] for i in rows]
    Nr = max(ys)//block + 1
    Nc = max(xs)//block + 1

    scores = (Xc[rows] @ W[:, pc_idx]).reshape(Nr, Nc)
    a = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    plt.figure(figsize=(5,4))
    plt.imshow(a, cmap="gray")
    plt.title(f"{filename} — PC{pc_idx+1} activation")
    plt.axis("off")
    out = os.path.join(RESULSTS_DIR+"/recon", f"{out_prefix}_{os.path.splitext(filename)[0]}_pc{pc_idx+1}_block{BLOCK}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out

# Reconstruction of one image from first k PCs
def reconstruct_image(filename, k=16, block=BLOCK, out_prefix="recon"):
    rows = [i for i, (fn, y, x) in enumerate(index) if fn == filename]
    assert rows, f"No patches found for {filename}"
    ys = [index[i][1] for i in rows]
    xs = [index[i][2] for i in rows]
    Nr = max(ys)//block + 1
    Nc = max(xs)//block + 1

    Wk = W[:, :k]
    Z = Xc[rows] @ Wk
    Xc_hat = Z @ Wk.T
    X_hat = Xc_hat + X.mean(axis=0, keepdims=True)

    recon = np.zeros((Nr*block, Nc*block), dtype=np.float32)
    t = 0
    for r in range(Nr):
        for c in range(Nc):
            recon[r*block:(r+1)*block, c*block:(c+1)*block] = X_hat[t].reshape(block, block)
            t += 1

    recon = np.clip(recon, 0.0, 1.0)
    plt.figure(figsize=(5,4))
    plt.imshow(recon, cmap="gray")
    plt.title(f"{filename} — Reconstruction with k={k} PCs")
    plt.axis("off")
    out = os.path.join(RESULSTS_DIR+"/recon", f"{out_prefix}_{os.path.splitext(filename)[0]}_k{k}_block{BLOCK}.png")
    plt.savefig(out, dpi=200); plt.close()
    plt.close()
    return out

#run
# plot_pc_filters(W, k=64, out_name=f"pc_filters_top64_{BLOCK}.png")
# plot_evr(EVR, out_name=f"evr_scree_block{BLOCK}.png")

example_file = index[80100][0] #0(1), 125100,21100(106), 300100,50100(117),500200,80100(129)
act_path = activation_map_for(example_file, pc_idx=0)
rec_path = reconstruct_image(example_file, k=32)

print("finished!")