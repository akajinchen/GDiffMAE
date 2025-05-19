import torch
import numpy as np
import inspect
import torch.nn.functional as F


def innerProduct(usrEmbeds, itmEmbeds):
	return torch.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2) / W.shape[0]
	return ret


def calcRegLoss_normal(*args):
	ret = 0
	for embeds in args:
		if inspect.isgenerator(embeds):
			for embed in embeds:
				ret += embed.norm(2)
		else:
			ret += embeds.norm(2) ** 2 / embeds.shape[0]
	return ret


def contrast(nodes, allEmbeds, allEmbeds2=None):
	if allEmbeds2 is not None:
		pckEmbeds = allEmbeds[nodes]
		scores = torch.log(torch.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()

	else:
		uniqNodes = torch.unique(nodes)
		pckEmbeds = allEmbeds[uniqNodes]
		scores = torch.log(torch.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
	return scores


def Uniformity_loss(idx1, emb1, idx2=None, emb2=None):
	if idx2 is None or emb2 is None:
		idx1 = torch.unique(idx1)
		emb = emb1[idx1]
		uniformity_loss = torch.log(torch.exp((emb @ emb1.T)).sum(dim=1)).mean()
	else:
		idx1 = torch.unique(idx1)
		idx2 = torch.unique(idx2)
		emb1 = emb1[idx1]
		uniformity_loss = torch.log(torch.exp((emb1 @ emb2.T)).sum(dim=1)).mean()
	return uniformity_loss

def Uniformity_loss1(idx1, emb1, idx2=None, emb2=None):
	if idx2 is None or emb2 is None:
		idx1 = torch.unique(idx1)
		emb = emb1[idx1]
		uniformity_loss = torch.log(torch.exp((emb @ emb.T)).sum(dim=1)).mean()
	else:
		idx1 = torch.unique(idx1)
		idx2 = torch.unique(idx2)
		emb1 = emb1[idx1]
		emb2 = emb2[idx2]
		uniformity_loss = torch.log(torch.exp((emb1 @ emb2.T)).sum(dim=1)).mean()
	return uniformity_loss


def Uniformity_loss2(emb1, emb2, t):
	emb1 = F.normalize(emb1, dim=-1)
	emb2 = F.normalize(emb2, dim=-1)
	uniformity_loss = torch.log(torch.exp(2.*t*(emb1@emb2.T)).mean())
	return uniformity_loss


def InfoNce(view1, view2, temperature: float):
	view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
	score = (view1 @ view2.T) / temperature
	score = -torch.diag(F.log_softmax(score, dim=1)).mean()
	return score

def kl_loss(embeds, mu, logvar, eps, k, j):
	SMALL = 1e-6
	std = torch.exp(0.5 * logvar)
	log_prior_ker = torch.sum(-0.5 * embeds.pow(2), dim=[1,2]).mean()
	mu_mix, mu_emb = mu[:k, :, :], mu[k:, :, :]
	std_mix, std_emb = std[:k, :, :], std[k:, :, :]
	Z = embeds.unsqueeze(1)
	mu_mix = mu_mix.unsqueeze(0)
	std_mix = std_mix.unsqueeze(0)
	log_post_ker_JK = - torch.sum(0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2,-1])
	log_post_ker_JK += -torch.sum((std_mix + SMALL).log(), dim=[-2,-1])
	log_post_ker_J = - torch.sum(0.5 * eps.pow(2), dim=[-2,-1])
	log_post_ker_J += - torch.sum((std_emb + SMALL).log(), dim=[-2,-1])
	log_post_ker_J = log_post_ker_J.view(-1,1)
	log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)
	log_post_ker -= np.log(k + 1.) / j
	log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()
	return log_prior_ker, log_posterior_ker





