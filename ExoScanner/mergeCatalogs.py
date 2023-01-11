# This file implements the algorithm described in
# "FOCAS Automatic Catalog Matching Algorithms".
# It finds which stars are the same in different catalogs.

import math
import numpy as np
from queue import PriorityQueue
from numba import jit, njit
from matplotlib import pyplot as plt


def convertToTriangleSpace(sides):
    sides.sort()
    return (sides[1] / sides[2], sides[0] / sides[2])


def selectBrightest(catalog, N):
    res = np.argpartition(catalog["flux"].data, -20, axis=None)[-20:]

    return res


def getDistances(points):
    distances = []
    for i in range(0, len(points)):
        distances.append([])
        for j in range(0, len(points)):
            distances[-1].append(math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2))

    return distances


def findTriangles(catalog, N):
    selection = selectBrightest(catalog, N)
    points = []
    for i in selection:
        points.append((catalog[i]["xcentroid"], catalog[i]["ycentroid"]))
    distances = getDistances(points)

    triangles = []
    for i in range(0, N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tris = convertToTriangleSpace([distances[i][j], distances[j][k], distances[k][i]])
                if tris[1] < 0.1:
                    continue  # ignore long triangles
                triangles.append((*tris, i, j, k))
    triangles = np.array(triangles)
    return triangles

@njit
def binarySearchTriangles(triangles, r_i, value):
    l = 0
    r = r_i

    while l != r:
        m = int((l + r) / 2)
        if triangles[m][0] < value:
            l = m + 1
        else:
            r = m

    return l


def firstMatchingUsingTriangles(mergedCatalog, newCatalog, N, epsilon=0.04):
    votes = np.zeros((N, N))
    oldTriangles = findTriangles(mergedCatalog, N)
    oldTriangles.sort()
    oldSelection = selectBrightest(mergedCatalog, N)
    triangles = findTriangles(newCatalog, N)
    selection = selectBrightest(newCatalog, N)
    r_i = len(oldTriangles) - 1
    # Optimize later with BS (if necessary...)
    for i in range(0, len(triangles)):
        for j in range(binarySearchTriangles(oldTriangles, r_i, triangles[i][0] - epsilon),
                       binarySearchTriangles(oldTriangles, r_i, triangles[i][0] + epsilon)):
            if (((triangles[i][0] - oldTriangles[j][0]) ** 2) + ((triangles[i][1] - oldTriangles[j][1]) ** 2)) < (
                    epsilon ** 2):
                votes[triangles[i].astype(int)[2:5], oldTriangles[j].astype(int)[2:5]] += 1

    matching = []
    indices = np.vstack(np.unravel_index(np.argpartition(votes.flatten(), -N)[-N:], votes.shape)).T
    for i in range(N):
        matching.append((selection[indices[i, 0]], oldSelection[indices[i, 1]]))

    return matching


def getTranslation(mergedCatalog, newCatalog, N):
    matching = firstMatchingUsingTriangles(mergedCatalog, newCatalog, N)

    translation = np.array([None])

    while True:
        newX = []
        newY = []
        mergedX = []
        mergedY = []

        for i in matching:
            newX.append(newCatalog[i[0]]["xcentroid"])
            newY.append(newCatalog[i[0]]["ycentroid"])
            mergedX.append(mergedCatalog[i[1]]["xcentroid"])
            mergedY.append(mergedCatalog[i[1]]["ycentroid"])
            if (
                    len(newX) == 6 and translation.any() == None): break  # Only use first 6 in first iteration to make sure it's precise enough

        newCoordinates = np.vstack([newX, newY, np.ones(len(newX))]).T
        mergedCoordinates = np.vstack([mergedX, mergedY]).T

        translation = np.linalg.lstsq(newCoordinates, mergedCoordinates, rcond=None)[0]

        distancesSquared = []
        for i in matching:
            transformedX, transformedY = [newCatalog[i[0]]["xcentroid"], newCatalog[i[0]]["ycentroid"], 1] @ translation
            distancesSquared.append(((mergedCatalog[i[1]]["xcentroid"] - transformedX) ** 2 + (
                    mergedCatalog[i[1]]["ycentroid"] - transformedY) ** 2, *i))

        distancesSquared.sort()

        sigma = distancesSquared[int(0.6 * len(distancesSquared))][0]

        newMatching = []

        for i in distancesSquared:
            if (i[0] <= sigma * 2):
                newMatching.append((i[1], i[2]))

        if (len(matching) == len(newMatching)):
            break

        matching = newMatching

    return translation


def binarySearchCatalog(table, value, translation):
    l = 0
    r = len(table) - 1

    while l != r:
        m = int((l + r) / 2)
        if ([table[m]["xcentroid"], table[m]["ycentroid"], 1] @ translation)[0] < value:
            l = m + 1
        else:
            r = m

    return l


def mergeNext(mergedCatalog, newCatalog, allowedError=3):
    translation = getTranslation(mergedCatalog, newCatalog, 20)

    newCatalog.sort("xcentroid")

    transition = []
    for i in range(len(mergedCatalog)):
        bestDist = allowedError * allowedError + 0.1
        index = -1
        for j in range(binarySearchCatalog(newCatalog, mergedCatalog[i]["xcentroid"] - allowedError, translation),
                       binarySearchCatalog(newCatalog, mergedCatalog[i]["xcentroid"] + allowedError, translation)):
            mergedX, mergedY = [mergedCatalog[i]["xcentroid"], mergedCatalog[i]["ycentroid"]]
            newX, newY = [newCatalog[j]["xcentroid"], newCatalog[j]["ycentroid"], 1] @ translation
            dist = (newX - mergedX) ** 2 + (newY - mergedY) ** 2
            if (dist < bestDist or bestDist == -1):
                bestDist = dist
                index = newCatalog[j]["id"] - 1

        bestDist = math.sqrt(bestDist)

        transition.append(index)

    # for i in range(len(newCatalog)):
    #     if (matched[i]): continue
    #     mergedCatalogWithNew.add_row(newCatalog[i])

    return transition


def mergeCatalogs(catalogs):
    transitions = []
    for i in range(1, len(catalogs)):
        transitions.append(mergeNext(catalogs[0], catalogs[i][:]))
        catalogs[i].sort("id")

    return transitions
