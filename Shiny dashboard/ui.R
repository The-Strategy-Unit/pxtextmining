library(tidyverse)
library(shinydashboard)
library(reactable)

test_data <- read.csv('../y_pred_and_x_test.csv')

body <- dashboardPage(
  dashboardHeader(title = "Patient feedback and its predicted label"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Widgets", tabName = "widgets", icon = icon("th"))
    )
  ),
  dashboardBody(
    # Boxes need to be put in a row (or column)
    fluidRow(
      column(7, 
             box(width = NULL,
                 textOutput("modelAccuracyBox"), background = 'red'
             )
      )
    ),
    
    fluidRow(
      column(width = 7,
             box(width = NULL,
                 selectInput("pred", "Choose a label:", 
                             choices=sort(unique(test_data$pred))),
                 reactableOutput("pedictedLabels"))),
      
      column(width = 5,
        box(plotOutput("tfidf_bars"), width = NULL),
        box(htmlOutput("tfidfExplanation"), background = 'red', width = NULL)
      )
    )
  )
)