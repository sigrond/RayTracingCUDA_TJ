//#include <sys/types.h>
//#include <sys/socket.h>
//#include <sys/un.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <WinSock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <cstdio>
//#include <unistd.h>
#include <cstdlib>
#include <exception>
#include<string>
#define DEFAULT_PORT "27015"
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")
using namespace std;
int main(int argc, char ** argv)
{
	try {
		int sock;
		SOCKET ConnectSocket = INVALID_SOCKET;
		struct sockaddr server;
		//sock = socket(AF_UNIX, SOCK_STREAM, 0);
		//if (sock < 0) {
		//	perror("opening stream socket");
		//	exit(1);
		//}
		//server.sun_family = AF_UNIX;
		//strcpy(server.sun_path, argv[1]);

		WSADATA wsaData;
		int iResult;
		// Initialize Winsock
		iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
		if (iResult != 0)
		{
			printf("WSAStartup failed: %d\n", iResult);
			return 1;
		}

		struct addrinfo *result = NULL, *ptr = NULL, hints;

		ZeroMemory(&hints, sizeof(hints));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;

		iResult = getaddrinfo("127.0.0.1", DEFAULT_PORT, &hints, &result);
		if (iResult != 0) {
			printf("getaddrinfo failed with error: %d\n", iResult);
			WSACleanup();
			return 1;
		}
		for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {
			ConnectSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
			if (ConnectSocket == INVALID_SOCKET)
			{
				printf("Error at socket(): %ld\n", WSAGetLastError());
				freeaddrinfo(result);
				WSACleanup();
				return 1;
			}


			iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
			if (iResult == SOCKET_ERROR) {
				closesocket(ConnectSocket);
				ConnectSocket = INVALID_SOCKET;
				continue;
			}
		}
		freeaddrinfo(result);
		char dummy[sizeof(int)];
		dummy[0] = stoi(argv[1]);
		iResult = send(ConnectSocket, dummy, 1 * sizeof(int), 0);
		if (iResult == SOCKET_ERROR) {
			printf("send failed with error: %d\n", WSAGetLastError());
			closesocket(ConnectSocket);
			WSACleanup();
			return 1;
		}
		do {

			iResult = recv(ConnectSocket, dummy, 1 * sizeof(int), 0);
			if (iResult > 0) {
				printf("Bytes received: %d\n", iResult);
				iResult = shutdown(ConnectSocket, SD_SEND);
				if (iResult == SOCKET_ERROR) {
					printf("shutdown failed with error: %d\n", WSAGetLastError());
					closesocket(ConnectSocket);
					WSACleanup();
					return 1;
				}
				iResult = 0;
			}
			else if (iResult == 0)
				printf("Connection closed\n");
			else
				printf("recv failed with error: %d\n", WSAGetLastError());

		} while (iResult > 0);
		closesocket(ConnectSocket);
		return 0;
	}
	catch (exception& e)
	{
		printf("exception: %s\n", e.what());
	}
	catch (...)
	{
		printf("unknown exception\n");
	}
}
