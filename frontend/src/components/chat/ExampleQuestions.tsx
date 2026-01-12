/**
 * Example questions accordion
 */

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Button } from '@/components/ui/button'
import { Lightbulb } from 'lucide-react'

const EXAMPLE_QUESTIONS = [
  'What are the main challenges for achieving net zero emissions by 2050?',
  'What is the projected global renewable energy capacity for 2028?',
  'How is the global EV market expected to evolve in the coming years?',
  'What percentage of global electricity generation came from renewables in 2023?',
  'What are the key policy recommendations for accelerating clean energy transitions?',
  'What is the projected investment in clean energy technologies for 2024?',
  'How do critical minerals supply chains impact renewable energy deployment?',
  'What are the exact CO2 emissions reduction targets in the Net Zero Roadmap?',
  'What role does energy efficiency play in reducing global energy demand?',
  'What are the specific challenges facing developing countries in clean energy transitions?',
]

interface ExampleQuestionsProps {
  onSelect: (question: string) => void
  disabled?: boolean
}

export function ExampleQuestions({ onSelect, disabled }: ExampleQuestionsProps) {
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="examples" className="border-none">
        <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
          <span className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4" />
            Example Questions
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <div className="flex flex-wrap gap-2 pt-2">
            {EXAMPLE_QUESTIONS.map((question) => (
              <Button
                key={question}
                variant="outline"
                size="sm"
                className="h-auto whitespace-normal text-left text-xs"
                onClick={() => onSelect(question)}
                disabled={disabled}
              >
                {question}
              </Button>
            ))}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
