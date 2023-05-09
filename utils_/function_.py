import torch
import inspect
from torch import optim
import numpy as np

def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets,meta in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)


def get_instance(module, class_name, *args, **kwargs):
    try:
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        return instance
    except AttributeError:
        available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]
        raise ValueError(f"{class_name} not found in the given module. Available classes: {', '.join(available_classes)}")


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def generate_negative_points(coordinates, width):
    # Sort the coordinates by y
    sorted_coordinates = sorted(coordinates, key=lambda x: x[1])

    negative_points = []

    # Iterate through each pair of points in the sorted list
    for i in range(len(sorted_coordinates) - 1):
        point1 = np.array(sorted_coordinates[i])
        point2 = np.array(sorted_coordinates[i + 1])

        # Compute the vector between the two points and normalize it
        direction = point2 - point1
        direction = direction / np.linalg.norm(direction)

        # Compute the perpendicular vector and normalize it
        perpendicular = np.array([-direction[1], direction[0]])

        # Calculate the midpoint between the two points
        midpoint = (point1 + point2) / 2

        # Calculate the two new points on the perpendicular line, with a distance of 'width' between them
        negative_point1 = midpoint + (width / 2) * perpendicular
        negative_point2 = midpoint - (width / 2) * perpendicular

        # Append the new points to the list of negative points
        negative_points.append(tuple(negative_point1))
        negative_points.append(tuple(negative_point2))

    return negative_points

