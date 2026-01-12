
import { sources } from "./sources"
import { typeSafeObjectEntries, typeSafeObjectFromEntries } from "./type.util"
import type { ColumnID, HiddenColumnID, Metadata, SourceID } from "./types"

export const columns = {
  papers: {
    zh: "论文列表",
  }
} as const

// We treat 'papers' as a fixed column for simplicity or let it be default
export const fixedColumnIds = ["papers"] as const satisfies Partial<ColumnID>[]
export const hiddenColumns = [] as HiddenColumnID[]

export const metadata: Metadata = typeSafeObjectFromEntries(typeSafeObjectEntries(columns).map(([k, v]) => {
    return [k, {
      name: v.zh,
      // Map sources that belong to this column
      sources: typeSafeObjectEntries(sources).filter(([, v]) => v.column === k).map(([k]) => k),
    }]
}))
