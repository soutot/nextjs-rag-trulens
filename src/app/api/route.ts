import {HNSWLib} from '@langchain/community/vectorstores/hnswlib'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {LangChainStream, StreamingTextResponse} from 'ai'
import {writeFile} from 'fs'
import {ConversationalRetrievalQAChain} from 'langchain/chains'
import {ChatMessageHistory, ConversationTokenBufferMemory} from 'langchain/memory'
import {ChatPromptTemplate} from 'langchain/prompts'
import {AIMessage, BaseMessage, ChainValues, HumanMessage} from 'langchain/schema'
import {z} from 'zod'

import {DEFAULT_OPENAI_MODEL, RAG_VECTOR_STORE_PATH} from '@/app/api/const'

const CONDENSE_QUESTION_TEMPLATE = `Given the following conversation and a follow up input, if it is a question rephrase it to be a standalone question.\n
  If it is not a question, just summarize the message. Give a response in the same language as the question.\n\n

  Chat history:\n
  {chat_history}
  \n\n
  Follow up input: {question}\n
  Standalone input:
  `
const QA_PROMPT_TEMPLATE = `You are a good assistant that answers questions. Your knowledge is strictly limited to the following pieces of context. Use it to answer the question at the end.
  If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
  If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
  Give a response in the same language as the question.
  Make sure to only answer the question, do not provide extra details that are not asked for.
  The context might contain pieces of diagrams or tables, so format your answer accordingly.\n\n

  Context: {context}

  Question: {question}
  Helpful answer in markdown:`

type ResponseProps = {
  messagesHistory: any[]
  prompt: string
}

const response = async ({messagesHistory, prompt}: ResponseProps) => {
  const baseMessages: BaseMessage[] = []

  messagesHistory.forEach((message) => {
    if (message.creator === 'USER') {
      baseMessages.push(new HumanMessage(message.text))
    }
    if (message.creator === 'AI') {
      baseMessages.push(new AIMessage(message.text))
    }
  })

  const chatHistory = new ChatMessageHistory(baseMessages)

  const {
    stream,
    handlers: {
      handleChainEnd,
      handleLLMStart,
      handleLLMNewToken,
      handleLLMError,
      handleChainStart,
      handleChainError,
      handleToolStart,
      handleToolError,
    },
  } = LangChainStream()

  let id = ''

  const handlers = {
    handleLLMStart: (llm: any, prompts: string[], runId: string) => {
      id = runId
      return handleLLMStart(llm, prompts, runId)
    },
    handleLLMNewToken,
    handleLLMError,
    handleChainStart,
    handleChainError,
    handleToolStart,
    handleToolError,
  }

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  })

  const vectorStore = await HNSWLib.load(RAG_VECTOR_STORE_PATH, embeddings)

  const retriever = vectorStore.asRetriever()
  const nonStreamLlm = new ChatOpenAI({
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: DEFAULT_OPENAI_MODEL,
  })

  const llm = new ChatOpenAI({
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
    streaming: true,
    modelName: DEFAULT_OPENAI_MODEL,
    callbacks: [handlers],
  })

  const memory = new ConversationTokenBufferMemory({
    llm,
    maxTokenLimit: 500,
    chatHistory,
    memoryKey: 'chat_history',
    outputKey: 'text',
    returnMessages: true,
  })

  const chain = ConversationalRetrievalQAChain.fromLLM(llm, retriever, {
    returnSourceDocuments: true,
    qaChainOptions: {
      type: 'stuff',
      prompt: ChatPromptTemplate.fromTemplate(QA_PROMPT_TEMPLATE),
    },
    memory,
    questionGeneratorChainOptions: {
      llm: nonStreamLlm,
      template: CONDENSE_QUESTION_TEMPLATE,
    },
  })

  chain.invoke({question: prompt}).then(async (response) => {
    try {
      const sources = response?.sourceDocuments?.map((doc: any) => doc.pageContent)?.filter(Boolean)
      const result = {
        prompt,
        response: response.text || response.response,
        context: sources?.length ? sources : undefined,
      }
  
      await writeFile(`${process.env.TRULENS_RESULT_FILE}`, JSON.stringify(result), () => {})
    } catch (e) {
      console.log('ERROR: ', e)
    }

    await handleChainEnd(null, id)
  })

  return new StreamingTextResponse(stream)
}

export async function POST(request: Request) {
  const body = await request.json()
  const bodySchema = z.object({
    prompt: z.string(),
    messagesHistory: z.array(z.any()),
  })

  const {prompt, messagesHistory} = bodySchema.parse(body)

  return response({
    messagesHistory,
    prompt,
  })
}
