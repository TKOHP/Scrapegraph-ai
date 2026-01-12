
import { createFileRoute } from "@tanstack/react-router"
import { Column } from "~/components/column"
import { useSyncSources } from "~/hooks/useSyncSources"

export const Route = createFileRoute("/")({
  component: IndexComponent,
})

function IndexComponent() {
  useSyncSources()
  return <Column id="papers" />
}
