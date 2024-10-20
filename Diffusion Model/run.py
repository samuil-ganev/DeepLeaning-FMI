import os.path

from parameters import *
from clip import *
from vae import *
from unet import *
from utils import *
from PIL import Image, ImageOps
from collections import defaultdict

import sys
import pickle

from generator import *

import glob
import pandas as pd
import requests
from io import BytesIO

import matplotlib.pyplot as plt

np.random.seed(0)
'''
with open('pairs.txt', 'r') as pairsFile:
   pairs = pairsFile.readlines()

   dictionary = defaultdict(int)
   text = ''

   for pair in pairs:
       t = pair.split('::')[1] + ' '
       text += t.lower()

   dictionary[padToken] = 1
   dictionary[unkToken] = 1
   for word in text.split():
       dictionary[word] += 1

   L = sorted([(token, dictionary[token]) for token in dictionary], key=lambda x: x[1], reverse=True)
   tokens = [token for token, _ in L]
   token2ind = {p: i for i, p in enumerate(tokens)}

pickle.dump((token2ind), open(tokensDataFile, 'wb'))
'''

class SD(nn.Module):

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, token2ind, n_steps=train_steps, min_beta=10 ** -4, max_beta=0.02):
        super().__init__()

        self.n_steps = n_steps
        self.betas = torch.linspace(min_beta, max_beta, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

        # self.vae_encoder = VAE_Encoder().to(device)
        self.clip = CLIP(token2ind, clip_heads_val, d_cross_val)
        # self.diff = Diffusion()
        # self.vae_decoder = VAE_Decoder().to(device)

        state_dict = load_from_standard_weights(model_file, device)

        self.vae_encoder = VAE_Encoder().to(device)
        self.vae_encoder.load_state_dict(state_dict['encoder'], strict=True)

        self.vae_decoder = VAE_Decoder().to(device)
        self.vae_decoder.load_state_dict(state_dict['decoder'], strict=True)

        self.diff = Diffusion().to(device)
        # self.diff.load_state_dict(state_dict['diffusion'], strict=True)

    def forward(self, x, y):
        self.train()
        x = rescale(x, (0.0, 255.0), (-1.0, 1.0))
        t = torch.randint(0, self.n_steps, (x.shape[0],)).to(device)

        with torch.no_grad():
            vae_out = self.vae_encoder(x)

        clip_out = self.clip(y)

        eta = torch.randn(vae_out.shape).to(device)
        x = self.add_noise(vae_out, t, eta)

        eta_out = self.diff(x, clip_out, get_time_embedding(t))

        mse = F.mse_loss(eta_out, eta)
        return mse

    def add_noise(self, x0, t, eta):
        a_bar = self.alpha_bars[t]

        noisy = a_bar.sqrt().reshape(x0.shape[0], 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(x0.shape[0], 1, 1, 1) * eta
        return noisy.to(device)


class VT(nn.Module):

    def save(self, fileName):
        torch.save(self.state_dict(), fileName + 'VT')

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, token2ind):
        super().__init__()

        min_beta, max_beta = 10 ** -4, 0.02
        self.n_steps = 100
        self.betas = torch.linspace(min_beta, max_beta, self.n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

        self.clip = CLIP(token2ind, clip_heads_val, d_cross_val)

        state_dict = load_from_standard_weights(model_file, device)

        self.vae_encoder = VAE_Encoder().to(device)
        self.vae_encoder.load_state_dict(state_dict['encoder'], strict=True)

        self.vae_decoder = VAE_Decoder().to(device)
        self.vae_decoder.load_state_dict(state_dict['decoder'], strict=True)

        self.vt = VisualTransformer(4, 49 * 4, d_cross_val, 8).to(device)
        # self.diff.load_state_dict(state_dict['diffusion'], strict=True)

    def forward(self, x, y):
        self.train()
        x = rescale(x, (0.0, 255.0), (-1.0, 1.0))
        t = torch.randint(0, 100, (x.shape[0],)).to(device)

        with torch.no_grad():
            vae_out = self.vae_encoder(x)

        clip_out = self.clip(y)

        eta = torch.randn(vae_out.shape).to(device)
        x = self.add_noise(vae_out, t, eta)

        eta_out = self.vt(x, clip_out)

        mse = F.mse_loss(eta_out, eta)
        return mse

    def add_noise(self, x0, t, eta):
        a_bar = self.alpha_bars[t]

        noisy = a_bar.sqrt().reshape(x0.shape[0], 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(x0.shape[0], 1, 1, 1) * eta
        return noisy.to(device)


def train(model, optimizer, bestMSE, mode = 'SD'):

    with open('pairs.txt', 'r') as pairsFile:
        pairs = pairsFile.readlines()

    avg_mse = bestMSE
    for epoch in range(epochs):  # направено е така че да работи само за batch = 1 !!!
        iter = 1
        last5MSE = [0] * 18
        for b in range(0, len(pairs), batch):
            img, tokens = pairs[b].split('::')
            tokens = [tokens.lower().strip().split()]
            img = torch.from_numpy(np.moveaxis(np.array(Image.open(img)), -1, 0).astype(np.float32)).unsqueeze(0).to(device)

            mse = model(img, tokens)
            print(f'Epoch {epoch}/{epochs} || Batch {iter}/{len(pairs)//batch} || MSE = {mse}')
            optimizer.zero_grad()
            mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            iter += 1

            last5MSE.pop()
            last5MSE.insert(0, mse)

        if avg_mse > sum(last5MSE) / len(last5MSE):
            if mode == 'SD':
                torch.save((avg_mse, optimizer.state_dict()), mse_osdFile + '.optim')
            else:
                torch.save((avg_mse, optimizer.state_dict()), mse_osdFile + 'VT' + '.optim')
            model.save(savedModelFileName)
            avg_mse = sum(last5MSE) / len(last5MSE)
            print('Model saved!')


# https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
# models = preload_models_from_standard_weights(model_file, device)


if len(sys.argv) > 1 and sys.argv[1] == 'prepare':

    pattern = '*.parquet'
    file_paths = glob.glob(pattern)

    dataframes = []

    padding = 2

    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df = combined_df[['info_src', 'image_description', 'info_alt']]

    file = open('pairs.txt', 'w', encoding='utf-8')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for row in combined_df.itertuples():
        try:
            image_url = row.info_src
            image_description = row.image_description
            info_alt = row.info_alt

            image_description = image_description.replace("\n", " ")
            info_alt = info_alt.replace("\n", " ")

            response = requests.get(image_url, headers=headers)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = ImageOps.expand(image, border=padding, fill='white')
                img_name = 'images\\' + image_url.split('/')[-1]
                if image.mode == 'RGB':
                    image.save(img_name)
                    if len(image_description) > 5:
                        file.write(img_name + '::' + image_description + "\n")
                    else:
                        file.write(img_name + '::' + info_alt + "\n")
        except:
            pass

    file.close()

    with open('pairs.txt', 'r') as pairsFile:
        pairs = pairsFile.readlines()

        dictionary = defaultdict(int)
        text = ''

        for pair in pairs:
            t = pair.split('::')[1] + ' '
            text += t.lower()


        dictionary[padToken] = 1
        dictionary[unkToken] = 1
        for word in text.split():
            dictionary[word] += 1

        L = sorted([(token, dictionary[token]) for token in dictionary], key=lambda x: x[1], reverse=True)
        tokens = [token for token, _ in L]
        token2ind = {p: i for i, p in enumerate(tokens)}

    pickle.dump((token2ind), open(tokensDataFile, 'wb'))

elif len(sys.argv) > 1 and sys.argv[1] == 'train' or sys.argv[1] == 'extratrain':
    mse = float('inf')
    token2ind = pickle.load(open(tokensDataFile, 'rb'))

    print(len(token2ind))
    mode = 'SD'

    if sys.argv[2] == 'SD':
        model = SD(token2ind)
    elif sys.argv[2] == 'VT':
        mode = 'VT'
        model = VT(token2ind)
    else:
        raise 'Не бе избран коректен модел!'

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=0.000000001)

    if sys.argv[1] == 'extratrain':
        if mode == 'SD':
            (mseSaved, osd) = torch.load(mse_osdFile + '.optim')
            model.load(savedModelFileName)
        else:
            (mseSaved, osd) = torch.load(mse_osdFile + 'VT' + '.optim')
            model.load(savedModelFileName + 'VT')
        mse = mseSaved
        optimizer.load_state_dict(osd)

    model.train()
    train(model, optimizer, mse, mode)

elif len(sys.argv) > 2 and sys.argv[1] == 'generate':
    prompt = ' '.join(sys.argv[3:])
    mode = sys.argv[2]

    token2ind = pickle.load(open(tokensDataFile, 'rb'))
    if mode == 'SD':
        model = SD(token2ind)
        model.load(savedModelFileName)
    else:
        model = VT(token2ind)
        model.load(f'{savedModelFileName}VT')

    image = generate(model, prompt, mode, device=device)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

elif len(sys.argv) > 2 and sys.argv[1] == 'fid':
    mode = sys.argv[2]

    token2ind = pickle.load(open(tokensDataFile, 'rb'))
    if mode == 'SD':
        model = SD(token2ind)
        model.load(savedModelFileName)
    else:
        model = VT(token2ind)
        model.load(f'{savedModelFileName}VT')

    img1 = torch.from_numpy(
        np.moveaxis(np.array(Image.open('images/jewish-girl-220x220.jpg')), -1, 0).astype(np.float32)).to(
        'cpu')
    img2 = torch.from_numpy(
        np.moveaxis(np.array(Image.open('images/frederic-sauvage-220x220.jpg')), -1, 0).astype(np.float32)).to(
        'cpu')
    img3 = torch.from_numpy(
        np.moveaxis(np.array(Image.open('images/dijon-saint-michel-220x220.jpg')), -1, 0).astype(np.float32)).to(
        'cpu')

    img1_gen = generate(model, 'Portrait of a Jewish girl living in Constantinople in the nineteenth century. The caption reads in the original French: Jeune fille juive. ', mode, device)
    img2_gen = generate(model, 'Alexis Sauvage (1786-1857), a French engineer and one of the inventors of screw-type marine propellers. The caption reads in the original French: Pierre-Louis-Frédéric Sauvage, inventeur de l’hélice, portrait d’après Gavarni. ', mode, device)
    img3_gen = generate(model, 'The construction of Saint Michael Church began in 1497 and it was consecrated in 1529, but the north tower wasnt finished until 1667. Originally conceived according to a gothic structure, it integrated Renaissance elements as the building process went on.  NB: The original caption mistakes Saint Michael Church for the Dijon cathedral. It reads: Cathédrale de Dijon. ', mode, device)

    print(fid_score(torch.stack([torch.from_numpy(img1_gen).to(dtype=torch.uint8), torch.from_numpy(img2_gen).to(dtype=torch.uint8), torch.from_numpy(img3_gen).to(dtype=torch.uint8)], dim=0), torch.stack([img1, img2, img3], dim=0)))