import { describe, expect, it } from 'vitest'

import { entityConfig, entityId, entityLabel, entityPath, findEntityRow, headToHeadPath, ratingsPagePath } from './entities'

const qbRows = [
  { qb_id: '00-0034857', qb_name: 'J.Allen', team: 'BUF' },
  { qb_id: '00-0039910', qb_name: null, team: 'WSH' },
]

describe('entityConfig', () => {
  it.each([
    ['teams', 'team_rating'],
    ['qbs', 'adj_qb_epa_per_dropback'],
  ] as const)('ranks the %s pages by %s', (kind, column) => {
    // Act
    const config = entityConfig(kind)

    // Assert
    expect(config.ratingColumn).toBe(column)
  })
})

describe('entityId and entityLabel', () => {
  it('identifies a quarterback by his player id', () => {
    // Act
    const id = entityId('qbs', qbRows[0])

    // Assert
    expect(id).toBe('00-0034857')
  })

  it('names a quarterback by his name', () => {
    // Act
    const label = entityLabel('qbs', qbRows[0])

    // Assert
    expect(label).toBe('J.Allen')
  })

  it('names a quarterback without a name by his id', () => {
    // Act
    const label = entityLabel('qbs', qbRows[1])

    // Assert
    expect(label).toBe('00-0039910')
  })

  it('names a team by its franchise name when the names are given', () => {
    // Act
    const label = entityLabel('teams', { team: 'WSH' }, { WSH: 'Washington Commanders' })

    // Assert
    expect(label).toBe('Washington Commanders')
  })
})

describe('findEntityRow', () => {
  it('finds a row by its id', () => {
    // Act
    const row = findEntityRow('qbs', qbRows, '00-0039910')

    // Assert
    expect(row?.team).toBe('WSH')
  })
})

describe('entityPath and ratingsPagePath', () => {
  it('links a detail page with its season', () => {
    // Act
    const path = entityPath('qbs', '00-0034857', 2025)

    // Assert
    expect(path).toBe('/qbs/00-0034857?season=2025')
  })

  it('links the head-to-head page of a pair', () => {
    // Act
    const path = headToHeadPath('qbs', 2025, '00-1', '00-2')

    // Assert
    expect(path).toBe('/head-to-head?kind=qbs&season=2025&a=00-1&b=00-2')
  })

  it.each([
    ['teams', 2024, '/teams?season=2024'],
    ['qbs', null, '/qbs'],
  ] as const)('links the %s index for season %s', (kind, season, expected) => {
    // Act
    const path = ratingsPagePath(kind, season)

    // Assert
    expect(path).toBe(expected)
  })
})
