
import { useQuery } from "@tanstack/react-query"
import { useSetAtom } from "jotai"
import { useEffect } from "react"
import { myFetch } from "~/utils"
import { currentSourcesAtom } from "~/atoms"
import type { SourceID } from "@shared/types"

export function useSyncSources() {
  const setSources = useSetAtom(currentSourcesAtom)

  const { data } = useQuery({
    queryKey: ["subscribe_from"],
    queryFn: async () => {
      const res = await myFetch("/subscribe_from")
      return res as string[]
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
  })

  useEffect(() => {
    if (data && data.length > 0) {
      // Assuming 'General' is the default hardcoded one, we merge or replace
      // If we want to replace the default 'general' with actual subscribe_from list
      // Note: 'general' in sources.json is lowercase. Backend returns potentially mixed case.
      // Let's filter out empty or null values
      const validSources = data.filter(Boolean) as SourceID[]
      
      // Update the atom
      setSources(validSources)
    }
  }, [data, setSources])
}
