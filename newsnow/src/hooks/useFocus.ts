
import type { SourceID } from "@shared/types"
import { useCallback, useMemo } from "react"

export function useFocus() {
  const toggleFocus = useCallback((id: SourceID) => {}, [])
  const isFocused = useCallback((id: SourceID) => false, [])

  return {
    toggleFocus,
    isFocused,
  }
}

export function useFocusWith(id: SourceID) {
  const toggleFocus = useCallback(() => {}, [])
  const isFocused = useMemo(() => false, [])

  return {
    toggleFocus,
    isFocused,
  }
}
