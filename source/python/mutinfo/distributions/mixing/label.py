import numpy
import math

from collections import defaultdict, Counter
from collections.abc import Sequence
from scipy.stats import randint
from scipy.stats._distn_infrastructure import rv_frozen, rv_discrete_frozen
from scipy.stats._multivariate import multi_rv_frozen
from typing import Any, Optional


class subsampler(multi_rv_frozen):
    """
    Frozen distribution with label data.
    """
    
    def __init__(
        self,
        data: Sequence,
        subset_indices: numpy.ndarray,
        replace: bool=False
    ) -> None:
        self.data = data
        self.subset_indices = subset_indices
        self.replace = replace
        

    def rvs(self, size: int=1) -> Sequence:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x : numpy.ndarray
            Random non-repetitive sampling.
        """
            
        length = len(self.subset_indices)
        if not self.replace:
            length = (int(size / length) + 1) * length
        
        indices = numpy.random.choice(length, size=size, replace=self.replace)
        indices = numpy.remainder(indices, len(self.subset_indices)) # TODO: come up with something more effective.
        
        return self.data[self.subset_indices[indices]]


def labeled_dataset_to_subsamplers(
    data: numpy.ndarray,
    labels: numpy.ndarray,
    split_by_labels: bool=True,
) -> dict[Any, subsampler]:
    """
    Convert labeled data into a dict of per-class subsamplers.

    Parameters
    ----------
    data : array_like
        Labeled data.
    labels : array_like

    Returns
    -------
    x : numpy.ndarray
        Random non-repetitive sampling.
    """

    if split_by_labels:
        # Shitty as hell implementation.
        subsamplers = {}
        for label in numpy.unique(labels):
            subsamplers[label] = subsampler(data, numpy.nonzero(labels == label)[0])
    
        return subsamplers
    else:
        return subsampler(data, numpy.arange(0, len(data)))
<<<<<<< HEAD

# TODO: does it belong here?
def torchvision_default_transform(x: numpy.ndarray, to_CHW: bool=False) -> numpy.ndarray:   
    x = x / 255
    if len(x.shape) < 4:
        x = x[:,None,...]
    
    if to_CHW:
        x = x.transpose((0, 3, 1, 2))

    return x

def embedding_with_resnet18_backbone(
    x: numpy.ndarray,
    checkpoint_path: Optional[str]=None,
    embeddings_dim: Optional[int]=None,
    backbone_name: str="resnet18",
) -> numpy.ndarray:
    """
    Extract embeddings from CIFAR-10 images using a trained ResNet backbone.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input images with shape (N, H, W, C) or (N, C, H, W).
        Values should be in range [0, 1] or [0, 255].
    checkpoint_path : str, optional
        Path to the checkpoint file. If None, tries common checkpoint locations.
    embeddings_dim : int, optional
        If provided, validates that the checkpoint matches this embedding dimension.
        (Legacy checkpoints only support 512.)
    backbone_name : str
        Torchvision backbone name (default: "resnet18").
    
    Returns
    -------
    embeddings : numpy.ndarray
        Feature embeddings with shape (N, embeddings_dim).
    """
    import torch
    import torch.nn as nn
    import torchvision
    from pathlib import Path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    if checkpoint_path is None:
        project_root = Path(__file__).resolve().parents[5]
        candidates = [
            project_root / 'resnet18_checkpoints' / 'best_model.pt',
            project_root / 'resnet18_checkpoints' / 'checkpoint_epoch_200.pt',
            project_root / 'checkpoints' / 'cifar10_supervised' / 'best_model.pt',
        ]
        for candidate in candidates:
            if candidate.exists():
                checkpoint_path = str(candidate)
                break
        else:
            raise FileNotFoundError(
                "No checkpoint found. Provide checkpoint_path, or create one in one of: "
                + ", ".join(str(c) for c in candidates)
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Prefer metadata if present
    ckpt_backbone_name = checkpoint.get('backbone_name') if isinstance(checkpoint, dict) else None
    if ckpt_backbone_name is not None and backbone_name == 'resnet18':
        backbone_name = ckpt_backbone_name

    ckpt_embeddings_dim = checkpoint.get('embeddings_dim') if isinstance(checkpoint, dict) else None
    ckpt_backbone_features = checkpoint.get('backbone_features') if isinstance(checkpoint, dict) else None

    has_embedding_head = isinstance(state_dict, dict) and 'embedding_head.weight' in state_dict
    if ckpt_embeddings_dim is None:
        if has_embedding_head:
            ckpt_embeddings_dim = int(state_dict['embedding_head.weight'].shape[0])
        elif 'backbone.fc.weight' in state_dict and int(state_dict['backbone.fc.weight'].shape[0]) == 10:
            # Legacy checkpoint: backbone.fc is the classifier (10 x 512)
            ckpt_embeddings_dim = int(state_dict['backbone.fc.weight'].shape[1])
        else:
            ckpt_embeddings_dim = 512

    if ckpt_backbone_features is None:
        if has_embedding_head:
            ckpt_backbone_features = int(state_dict['embedding_head.weight'].shape[1])
        elif 'backbone.fc.weight' in state_dict and int(state_dict['backbone.fc.weight'].shape[0]) == 10:
            ckpt_backbone_features = int(state_dict['backbone.fc.weight'].shape[1])

    if embeddings_dim is not None and int(embeddings_dim) != int(ckpt_embeddings_dim):
        raise ValueError(
            f"Checkpoint embeddings_dim={ckpt_embeddings_dim} does not match requested embeddings_dim={embeddings_dim}."
        )
    
    # Adapt backbone for CIFAR-10 (from supervised_learning_cifar10.py)
    def adapt_backbone_to_CIFAR(backbone):
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        backbone.maxpool = torch.nn.Identity()
        return backbone
    
    # Create backbone
    backbone = getattr(torchvision.models, backbone_name)()
    backbone = adapt_backbone_to_CIFAR(backbone)
    backbone_features = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()

    if ckpt_backbone_features is not None and int(ckpt_backbone_features) != backbone_features:
        raise ValueError(
            f"Checkpoint backbone_features={ckpt_backbone_features} does not match {backbone_name} features={backbone_features}."
        )

    # Load backbone weights - strip "backbone." prefix from checkpoint keys
    backbone_sd = {
        k.replace('backbone.', ''): v
        for k, v in state_dict.items()
        if k.startswith('backbone.') and not k.startswith('backbone.fc.')
    }
    backbone.load_state_dict(backbone_sd, strict=False)

    # Optional learned projection head (new checkpoints)
    if has_embedding_head:
        embedding_head = nn.Linear(backbone_features, int(ckpt_embeddings_dim))
        embedding_head.load_state_dict(
            {
                k.replace('embedding_head.', ''): v
                for k, v in state_dict.items()
                if k.startswith('embedding_head.')
            },
            strict=True,
        )
    else:
        embedding_head = nn.Identity()

    class _Embedder(nn.Module):
        def __init__(self, backbone, embedding_head):
            super().__init__()
            self.backbone = backbone
            self.embedding_head = embedding_head

        def forward(self, x):
            return self.embedding_head(self.backbone(x))

    model = _Embedder(backbone, embedding_head).to(device)
    model.eval()
=======

# TODO: does it belong here?
def torchvision_default_transform(x: numpy.ndarray, to_CHW: bool=False) -> numpy.ndarray:   
    x = x / 255
    if len(x.shape) < 4:
        x = x[:,None,...]
    
    if to_CHW:
        x = x.transpose((0, 3, 1, 2))

    return x

def embedding_with_resnet18_backbone(x: numpy.ndarray, checkpoint_path: str=None) -> numpy.ndarray:
    """
    Extract embeddings from CIFAR-10 images using a trained ResNet18 backbone.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input images with shape (N, H, W, C) or (N, C, H, W).
        Values should be in range [0, 1] or [0, 255].
    checkpoint_path : str, optional
        Path to the checkpoint file. If None, uses the best model from resnet18_checkpoints/.
    
    Returns
    -------
    embeddings : numpy.ndarray
        Feature embeddings with shape (N, 512) for ResNet18.
    """
    import torch
    import torch.nn as nn
    import torchvision
    from pathlib import Path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent.parent.parent.parent / 'resnet18_checkpoints' / 'best_model.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Adapt backbone for CIFAR-10 (from supervised_learning_cifar10.py)
    def adapt_backbone_to_CIFAR(backbone):
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        backbone.maxpool = torch.nn.Identity()
        return backbone
    
    # Create model
    backbone = torchvision.models.resnet18(num_classes=128)
    backbone = adapt_backbone_to_CIFAR(backbone)
    backbone.fc = nn.Linear(512, 10)  # Match the saved model structure
    
    # Load weights - strip "backbone." prefix from checkpoint keys
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    backbone.load_state_dict(state_dict)
    
    # Remove final classification layer to extract embeddings
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()
>>>>>>> remotes/upstream/mutinfo-minde
    
    # Prepare input
    if x.max() > 1.0:
        x = x / 255.0
    
    # Convert to CHW format if needed
    if x.shape[-1] == 3:  # HWC format
        x = x.transpose((0, 3, 1, 2))
    
    # Normalize using CIFAR-10 statistics
    mean = numpy.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = numpy.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    x = (x - mean) / std
    
    # Convert to tensor
    x_tensor = torch.from_numpy(x).float().to(device)
    
    # Extract embeddings
    with torch.no_grad():
<<<<<<< HEAD
        embeddings = model(x_tensor)
=======
        embeddings = backbone(x_tensor)
>>>>>>> remotes/upstream/mutinfo-minde
    
    return embeddings.cpu().numpy()

# TODO: does it belong here?
def torchvision_labeled_dataset_to_subsamplers(
    dataset,
    transform=torchvision_default_transform,
    split_by_labels: bool=True,
) -> dict[Any, subsampler]:
    """
    Convert torchvision labeled dataset into a dict of per-class subsamplers.

    Parameters
    ----------
    data : array_like
        Labeled data.
    labels : array_like

    Returns
    -------
    x : numpy.ndarray
        Random non-repetitive sampling.
    """
    
    return labeled_dataset_to_subsamplers(
        transform(numpy.asarray(dataset.data)),
        numpy.asarray(dataset.targets),
        split_by_labels
    )


class mixed_by_label(multi_rv_frozen):
    def __init__(
        self,
        marginal_distributions: dict[Any, list[multi_rv_frozen | rv_frozen]],
        labels_distribution: rv_discrete_frozen
    ) -> None:
        self._marginal_distributions = marginal_distributions
        self._labels_distribution = labels_distribution

    def rvs(self, size: int=1) -> list:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x_1, ..., x_k : numpy.ndarray
            Random sampling.
        """

        labels_tuple = self._labels_distribution.rvs(size=size)
        
        # Cannot come up with a better way to do this.
        sampling = []
        for labels, marginal_distribution in zip(labels_tuple, self._marginal_distributions):
            labels_counts = Counter(labels)
            labels_invargsort = numpy.empty_like(labels)
            labels_invargsort[numpy.argsort(labels, kind="stable")] = numpy.arange(len(labels))

            sampling.append(
                numpy.concatenate(
                    [
                        marginal_distribution[label].rvs(size=labels_counts[label])
                        for label in sorted(labels_counts.keys())
                    ],
                    axis=0
                )[labels_invargsort]
            )

        return tuple(sampling)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """

        if len(self._marginal_distributions) != 2:
            raise ValueError("Mutual information is only defined for pairs of random variables.")
        
        return self._labels_distribution.mutual_information