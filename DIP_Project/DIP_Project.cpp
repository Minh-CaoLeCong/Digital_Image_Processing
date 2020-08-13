// DIP_Project.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "DIP_Project.h"
#include <commdlg.h>
#include "../DIP_Library/DIP_Chapter3_Intensity_Transformations_and_Spatial_Filtering.h"

using namespace DIP;

#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Place code here.

    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_DIPPROJECT, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_DIPPROJECT));

    MSG msg;

    // Main message loop:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_DIPPROJECT));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_DIPPROJECT);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	static OPENFILENAME ofn;       // common dialog box structure
	static char szFile[260];       // buffer for file name
	static Mat imgin;
	static Mat imgout;
    switch (message)
    {
	case WM_CREATE:

		// Initialize OPENFILENAME
		ZeroMemory(&ofn, sizeof(ofn));
		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = hWnd;
		ofn.lpstrFile = szFile;
		// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
		// use the contents of szFile to initialize itself.
		ofn.lpstrFile[0] = '\0';
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = "Images\0*.jpg;*.bmp;*.tif;*.gif;*.png\0";
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = NULL;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

		break;
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {

			//FILE
			case ID_FILE_OPENIMAGE:
				if (GetOpenFileName(&ofn) == TRUE)
				{
					imgin = imread(ofn.lpstrFile, CV_LOAD_IMAGE_ANYCOLOR);
					namedWindow("Image In", WINDOW_AUTOSIZE);
					imshow("Image In", imgin);
				}
				break;
			case ID_FILE_SAVEIMAGE:
				if (GetSaveFileName(&ofn) == TRUE)
					imwrite(ofn.lpstrFile, imgout);
				break;

			//CHAPTER3
			case ID_CHAPTER3_RGB2GRAY:
				cvtColor(imgin, imgin, CV_RGB2GRAY);
				imgout = imgin.clone();
				namedWindow("RGB2GRAY", WINDOW_AUTOSIZE);
				imshow("RGB2GRAY", imgout);
				break;

			case ID_CHAPTER3_NEGATIVE:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::Negative(imgin, imgout);
				namedWindow("Negative", WINDOW_AUTOSIZE);
				imshow("Negative", imgout);
				break;

			case ID_CHAPTER3_LOGARIT:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::Logarit(imgin, imgout);
				namedWindow("Logarit", WINDOW_AUTOSIZE);
				imshow("Logarit", imgout);
				break;

			case ID_CHAPTER3_HISTOGRAM:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::Histogram(imgin, imgout);
				namedWindow("Histogram", WINDOW_AUTOSIZE);
				imshow("Histogram", imgout);
				break;

			case ID_CHAPTER3_HISTOGRAMRGB:
				DIP_Chapter3::Histogram_RGB(imgin);
				break;

			case ID_CHAPTER3_HISTOGRAMEQUALIZATION:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::HistogramEqualization(imgin, imgout);
				namedWindow("HistogramEqualization", WINDOW_AUTOSIZE);
				imshow("HistogramEqualization", imgout);
				break;

			case ID_CHAPTER3_POWER:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::Power(imgin, imgout);
				namedWindow("Power", WINDOW_AUTOSIZE);
				imshow("Power", imgout);
				break;

			case ID_CHAPTER3_WHITEBALANCING:
				/*imgout = Mat(imgin.size(), CV_8UC3);*/
				/*Mat imgout;*/
				imgout = imgin.clone();
				DIP_Chapter3::GreyWorld_Algorithm(imgin, imgout);
				namedWindow("GreyWorld_Algorithm", WINDOW_AUTOSIZE);
				imshow("GreyWorld_Algorithm", imgout);
				break;

			case ID_CHAPTER3_EQUALIZE:
				imgout = imgin.clone();
				DIP_Chapter3::equalizeIntensity(imgin, imgout);
				namedWindow("Equalize_Intensity", WINDOW_AUTOSIZE);
				imshow("Equalize_Intensity", imgout);
				break;

			case ID_CHAPTER3_PIECEWISELINEAR:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::PiecewiseLinear(imgin, imgout);
				namedWindow("PiecewiseLinear", WINDOW_AUTOSIZE);
				imshow("PiecewiseLinear", imgout);
				break;

			case ID_CHAPTER3_LOCALHISTOGRAM:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::LocalHistogram(imgin, imgout);
				namedWindow("LocalHistogram", WINDOW_AUTOSIZE);
				imshow("LocalHistogram", imgout);
				break;

			case ID_CHAPTER3_MYFILTER2D:
				imgout = Mat(imgin.size(), CV_8UC1);
				DIP_Chapter3::MyFilter2D(imgin, imgout);
				namedWindow("MyFilter2D", WINDOW_AUTOSIZE);
				imshow("MyFilter2D", imgout);
				break;

			//HELP
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code that uses hdc here...
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
