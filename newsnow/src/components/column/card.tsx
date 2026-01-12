
import type { NewsItem, SourceID } from "@shared/types"
import { useQuery } from "@tanstack/react-query"
import { useInView } from "framer-motion"
import { forwardRef, useImperativeHandle, useRef } from "react"
import { OverlayScrollbar } from "../common/overlay-scrollbar"
import { myFetch } from "~/utils"
import { useRefetch } from "~/hooks/useRefetch"
import { sources } from "@shared/sources"

export interface ItemsProps extends React.HTMLAttributes<HTMLDivElement> {
  id: SourceID
  isDragging?: boolean
  setHandleRef?: (ref: HTMLElement | null) => void
}

interface NewsCardProps {
  id: SourceID
  setHandleRef?: (ref: HTMLElement | null) => void
}

export const CardWrapper = forwardRef<HTMLElement, ItemsProps>(({ id, isDragging, setHandleRef, style, ...props }, dndRef) => {
  const ref = useRef<HTMLDivElement>(null)

  const inView = useInView(ref, {
    once: true,
  })

  useImperativeHandle(dndRef, () => ref.current! as HTMLDivElement)

  // Fallback color if source not found
  const sourceColor = sources[id]?.color || "blue"

  return (
    <div
      ref={ref}
      className={$(
        "flex flex-col h-500px rounded-2xl p-4 cursor-default",
        "transition-opacity-300",
        isDragging && "op-50",
        `bg-${sourceColor}-500 dark:bg-${sourceColor} bg-op-40!`,
      )}
      style={{
        transformOrigin: "50% 50%",
        ...style,
      }}
      {...props}
    >
      {inView && <NewsCard id={id} setHandleRef={setHandleRef} />}
    </div>
  )
})

function NewsCard({ id, setHandleRef }: NewsCardProps) {
  const { refresh } = useRefetch()
  
  // Mapping logic: if ID is "general", we fetch "General" or all. 
  // If we had multiple subjects, we would pass 'id' as 'subject'.
  const subjectParam = id === "general" ? undefined : id

  const { data, isFetching, isError } = useQuery({
    queryKey: ["source", id, refresh], // Add refresh to trigger refetch
    queryFn: async () => {
      const res: any = await myFetch("/papers", {
        query: { subscribe_from: subjectParam }
      })
      // API returns { items: [...] }
      return res
    },
    staleTime: 1000 * 60 * 5, // 5 mins
    refetchOnMount: false,
    refetchOnReconnect: false,
  })

  const sourceName = sources[id]?.name || id

  return (
    <div className="h-full flex flex-col">
       {/* Header */}
       <div 
         ref={setHandleRef} 
         className="flex items-center gap-2 mb-2 cursor-grab active:cursor-grabbing"
       >
         <span className="font-bold text-lg">{sourceName}</span>
         {isFetching && <div className="i-svg-spinners-ring-resize" />}
       </div>
       
       {isError ? (
           <div className="flex-1 flex items-center justify-center text-red-500">
               加载失败
           </div>
       ) : (
           <OverlayScrollbar className="flex-1 overflow-y-auto">
             <div className="flex flex-col gap-3 pb-2">
               {data?.items?.map((item: NewsItem) => (
                 <PaperItem key={item.id} item={item} />
               ))}
               {data?.items?.length === 0 && (
                   <div className="text-center text-sm opacity-50 py-4">暂无论文</div>
               )}
             </div>
           </OverlayScrollbar>
       )}
    </div>
  )
}

function PaperItem({ item }: { item: NewsItem }) {
    return (
        <div className="bg-white/60 dark:bg-black/20 p-3 rounded-xl hover:bg-white/90 dark:hover:bg-black/40 transition flex flex-col gap-2 shadow-sm">
            <a href={item.url} target="_blank" className="font-medium text-sm leading-tight hover:text-blue-600 dark:hover:text-blue-400 break-words">
                {item.title}
            </a>
            
            <div className="flex items-center justify-between text-xs opacity-60">
                <span>{item.pubDate}</span>
                <span>{item.source}</span>
            </div>

            <div className="flex flex-wrap gap-2 text-xs mt-1">
                {item.pdfLink && (
                    <a href={item.pdfLink} target="_blank" className="px-2 py-0.5 bg-red-100/50 dark:bg-red-900/30 text-red-600 dark:text-red-300 rounded hover:bg-red-200 dark:hover:bg-red-900/50 transition">
                        PDF
                    </a>
                )}
                {item.overviewLink && (
                    <a href={item.overviewLink} target="_blank" className="px-2 py-0.5 bg-green-100/50 dark:bg-green-900/30 text-green-600 dark:text-green-300 rounded hover:bg-green-200 dark:hover:bg-green-900/50 transition">
                        总结
                    </a>
                )}
                {item.analysisLink && (
                    <a href={item.analysisLink} target="_blank" className="px-2 py-0.5 bg-purple-100/50 dark:bg-purple-900/30 text-purple-600 dark:text-purple-300 rounded hover:bg-purple-200 dark:hover:bg-purple-900/50 transition">
                        分析
                    </a>
                )}
            </div>
        </div>
    )
}
